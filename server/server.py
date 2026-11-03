"""Temporary drafts + named saved studies. Run uvicorn with --workers 1 only."""
import io, os, re, time, json, uuid, sqlite3, threading, hashlib, shutil, zipfile, logging
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
import numpy as np
import SimpleITK as sitk
import pydicom
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import Response

BASE=Path(__file__).resolve().parent
DATA=Path(os.getenv('DATA_DIR',str(BASE/'data'))).resolve()
TEMP=Path(os.getenv('TEMP_DIR',str(DATA.with_name(DATA.name+'_temp')))).resolve()
if DATA==TEMP or DATA in TEMP.parents or TEMP in DATA.parents:
    raise RuntimeError('DATA_DIR и TEMP_DIR должны быть разными соседними хранилищами, не вложенными друг в друга.')
DRAFTS=TEMP/'drafts'; RESULTS=TEMP/'results'
for path in (DATA,DRAFTS,RESULTS): path.mkdir(parents=True,exist_ok=True)
MODEL=Path(os.getenv('MODEL_PATH',str(BASE/'best_model_densenet201.pth')))
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_BYTES=int(os.getenv('MAX_UPLOAD_MB','2048'))*1024*1024
TTL=max(60,float(os.getenv('DRAFT_TTL_HOURS','24'))*3600)
CLEAN_INTERVAL=max(10,float(os.getenv('TEMP_CLEAN_INTERVAL_SECONDS','600')))
POOL=ThreadPoolExecutor(max_workers=1)
LOCK=threading.RLock(); JOBS={}; ENGINE=None; MODEL_ERROR=None; STOP=threading.Event()
LOG=logging.getLogger('liver.storage')

@contextmanager
def db():
    connection=sqlite3.connect(DATA/'studies.sqlite',timeout=30)
    connection.row_factory=sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()

with db() as connection:
    connection.execute('CREATE TABLE IF NOT EXISTS studies (id TEXT PRIMARY KEY, name TEXT, metadata TEXT, saved INTEGER DEFAULT 0, revision INTEGER DEFAULT 0, report TEXT DEFAULT "", metrics TEXT, model TEXT, updated TEXT)')
    columns={r['name'] for r in connection.execute('PRAGMA table_info(studies)')}
    if 'storage_name' not in columns: connection.execute('ALTER TABLE studies ADD COLUMN storage_name TEXT')
    if 'original_name' not in columns: connection.execute('ALTER TABLE studies ADD COLUMN original_name TEXT')
    connection.execute('CREATE UNIQUE INDEX IF NOT EXISTS unique_storage_name ON studies(storage_name COLLATE NOCASE) WHERE storage_name IS NOT NULL')


def now_iso(): return datetime.now(timezone.utc).isoformat()
def validate_id(value):
    if not re.fullmatch(r'[a-f0-9]{32}',value or ''): raise HTTPException(400,'Некорректный идентификатор')
    return value

def session_id(value):
    if not re.fullmatch(r'[a-f0-9]{32}',value or ''):
        raise HTTPException(400,'Требуется обновлённый клиент: отсутствует X-Client-Session')
    return value

def atomic_json(path,value):
    temporary=path.with_name(path.name+'.new')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,path)

def saved_row(sid):
    validate_id(sid)
    with db() as connection: r=connection.execute('SELECT * FROM studies WHERE id=?',(sid,)).fetchone()
    return dict(r) if r else None

def child_path(root,name):
    p=(root/name).resolve()
    if p.parent!=root.resolve(): raise HTTPException(400,'Недопустимый путь')
    return p

def lookup(sid,owner=None):
    validate_id(sid)
    r=saved_row(sid)
    if r:
        # Legacy records stay intact; their files still live under data/<id>.
        return r,child_path(DATA,r.get('storage_name') or sid),False
    path=DRAFTS/sid; manifest=path/'draft.json'
    if not manifest.exists(): raise HTTPException(404,'Черновик не найден или срок хранения истёк')
    r=json.loads(manifest.read_text(encoding='utf-8'))
    if owner is not None and r['owner']!=owner: raise HTTPException(403,'Черновик принадлежит другому сеансу')
    return r,path,True

def active(sid,owner=None):
    return any(j['study_id']==sid and j['status'] in ('queued','running') and (owner is None or j['owner']==owner) for j in JOBS.values())

def touch_draft(path,record,force=False):
    if force or time.time()-record.get('last_seen',0)>60:
        record['last_seen']=time.time(); atomic_json(path/'draft.json',record)

def result_path(owner,sid): return RESULTS/owner/sid

def touch_session(owner):
    with LOCK:
        for manifest in DRAFTS.glob('*/draft.json'):
            try:
                r=json.loads(manifest.read_text(encoding='utf-8'))
                if r['owner']==owner: touch_draft(manifest.parent,r)
            except (OSError,ValueError,KeyError): LOG.exception('Не удалось продлить черновик %s',manifest)
        for manifest in (RESULTS/owner).glob('*/result.json'):
            try:
                r=json.loads(manifest.read_text(encoding='utf-8'))
                if time.time()-r.get('last_seen',0)>60:
                    r['last_seen']=time.time(); atomic_json(manifest,r)
            except (OSError,ValueError): LOG.exception('Не удалось продлить результат %s',manifest)


def validate_folder_name(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name or ''):
        raise HTTPException(400,'Название: 1–64 символа, латинские буквы, цифры, _ и -. Первый символ — буква или цифра.')
    reserved={'CON','PRN','AUX','NUL'}|{f'{p}{i}' for p in ('COM','LPT') for i in range(1,10)}
    if name.upper() in reserved: raise HTTPException(400,'Это имя зарезервировано Windows')
    return name

def check_name_available(name):
    name=validate_folder_name(name)
    if any(p.name.casefold()==name.casefold() for p in DATA.iterdir()):
        raise HTTPException(409,'Такое имя папки уже занято. Введите другое.')
    with db() as connection:
        if connection.execute('SELECT 1 FROM studies WHERE storage_name=? COLLATE NOCASE',(name,)).fetchone():
            raise HTTPException(409,'Это название уже используется')
    return name


def metrics(mask,spacing):
    voxel=float(np.prod(spacing))/1000
    healthy=int(np.count_nonzero(mask==1))*voxel
    tumor=int(np.count_nonzero(mask==2))*voxel
    total=healthy+tumor
    return dict(healthy=healthy,total=total,tumor=tumor,percent=100*tumor/total if total else None)


def recover_interrupted_first_saves():
    # A moved draft is recoverable if a crash happened before the SQLite commit.
    with LOCK:
        for manifest in DATA.glob('*/draft.json'):
            try:
                r=json.loads(manifest.read_text(encoding='utf-8')); sid=validate_id(r['id'])
                existing=saved_row(sid)
                if existing and existing['saved']:
                    manifest.unlink()
                elif existing is None:
                    target=DRAFTS/sid
                    if not target.exists():
                        os.replace(manifest.parent,target)
                        LOG.warning('Восстановлен незавершённый черновик %s',sid)
            except Exception: LOG.exception('Не удалось восстановить %s; оставлено без удаления',manifest)


def cleanup_expired():
    cutoff=time.time()-TTL
    with LOCK:
        for manifest in list(DRAFTS.glob('*/draft.json'))+list(RESULTS.glob('*/*/result.json')):
            try:
                r=json.loads(manifest.read_text(encoding='utf-8'))
                sid=r['id']; owner=r['owner']
                if r.get('last_seen',time.time())<cutoff and not active(sid,owner):
                    # Never remove a folder that has already been adopted as saved data.
                    shutil.rmtree(manifest.parent)
                    LOG.info('Удалён просроченный временный объект %s',manifest.parent)
            except Exception: LOG.exception('Ошибка очистки %s',manifest)
        # Failed/interrupted uploads without a ready manifest.
        for path in DRAFTS.iterdir():
            if path.is_dir() and not (path/'draft.json').exists() and path.stat().st_mtime<cutoff and not active(path.name):
                shutil.rmtree(path,ignore_errors=True)
        for jid in list(JOBS):
            j=JOBS[jid]
            if j['status'] in ('done','error') and j.get('finished',time.time())<cutoff: JOBS.pop(jid,None)


def cleanup_loop():
    while not STOP.wait(CLEAN_INTERVAL): cleanup_expired()

class DenseNetBackend:
    def __init__(self):
        self.net=smp.Unet(encoder_name='densenet201',encoder_weights=None,in_channels=1,classes=3)
        self.net.load_state_dict(torch.load(MODEL,map_location='cpu',weights_only=True))
        self.net.to(DEVICE).eval()
        self.version='densenet201:'+hashlib.sha256(MODEL.read_bytes()).hexdigest()[:16]
    @torch.inference_mode()
    def predict(self, image, progress):
        volume=sitk.GetArrayFromImage(image).astype(np.float32)
        mask=np.zeros(volume.shape,np.uint8)
        # Native slice order: no orientation rotation and no Z resampling.
        for z,s in enumerate(volume):
            x=torch.from_numpy(np.clip((s+150)/400,0,1).copy())[None,None].to(DEVICE)
            x=F.interpolate(x,size=(256,256),mode='bilinear',align_corners=False,antialias=True)
            logits=self.net(x); pred=logits.argmax(1)
            pred[logits.softmax(1)[:,2]>.3]=2
            mask[z]=F.interpolate(pred[:,None].float(),size=s.shape,mode='nearest')[0,0].cpu().numpy().astype(np.uint8)
            progress(int(100*(z+1)/len(volume)))
        return mask

def validate_image(image):
    if image.GetDimension()!=3 or image.GetNumberOfComponentsPerPixel()!=1: raise ValueError('Нужно скалярное 3D-изображение')
    sp=np.array(image.GetSpacing()); d=np.array(image.GetDirection()).reshape(3,3)
    if not np.all(np.isfinite(sp)) or np.any(sp<=0): raise ValueError('Некорректный spacing')
    if not np.allclose(d.T@d,np.eye(3),atol=1e-4): raise ValueError('Неортогональная сетка не поддерживается')
    if np.prod(image.GetSize())>512*512*2500: raise ValueError('Слишком большой объём для этой версии')
    if not np.isfinite(sitk.GetArrayViewFromImage(image)).all(): raise ValueError('В КТ есть NaN/Inf')

def load_source(path, dest):
    meta={}
    if path.name.endswith('.zip'):
        extracted=dest/'dicom'; extracted.mkdir()
        with zipfile.ZipFile(path) as archive:
            if sum(i.file_size for i in archive.infolist())>MAX_BYTES: raise ValueError('Распакованный архив слишком большой')
            for i,item in enumerate(archive.infolist()):
                if not item.is_dir():
                    with archive.open(item) as src, open(extracted/f'{i:06d}.dcm','wb') as out: shutil.copyfileobj(src,out)
        ids=sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(extracted)) or []
        if len(ids)!=1: raise ValueError(f'В архиве должно быть ровно 1 DICOM-серия; найдено {len(ids)}')
        names=sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(extracted),ids[0])
        ds=[pydicom.dcmread(n,stop_before_pixels=True) for n in names]
        if len(ds)<2: raise ValueError('Нужна серия из двух и более срезов')
        ori=np.asarray(ds[0].ImageOrientationPatient,float); normal=np.cross(ori[:3],ori[3:])
        pos=[]
        for d in ds:
            if int(getattr(d,'NumberOfFrames',1))!=1: raise ValueError('Enhanced/multiframe DICOM пока не поддерживается')
            if not np.allclose(np.asarray(d.ImageOrientationPatient,float),ori,atol=1e-4): raise ValueError('Ориентация срезов неодинакова')
            if not np.allclose(np.asarray(d.PixelSpacing,float),np.asarray(ds[0].PixelSpacing,float)): raise ValueError('Неодинаковый PixelSpacing')
            pos.append(np.asarray(d.ImagePositionPatient,float))
        delta=np.diff(pos,axis=0); dz=delta@normal
        if np.any(np.abs(dz)<1e-4) or not np.allclose(dz,dz[0],rtol=.01,atol=.01): raise ValueError('Неравномерная сетка или дубли срезов')
        if np.max(np.linalg.norm(delta-dz[:,None]*normal,axis=1))>.05: raise ValueError('Наклон gantry/shear пока не поддерживается')
        reader=sitk.ImageSeriesReader(); reader.SetFileNames(names); image=reader.Execute()
        for key in ['PatientID','StudyDate','Modality','Manufacturer','SeriesInstanceUID']:
            meta[key]=str(getattr(ds[0],key,''))
        meta['format']='DICOM'
    else:
        image=sitk.ReadImage(str(path)); meta['format']='NIfTI'
    validate_image(image)
    return image,meta

@asynccontextmanager
async def lifespan(app):
    global ENGINE,MODEL_ERROR
    recover_interrupted_first_saves(); cleanup_expired()
    try: ENGINE=DenseNetBackend()
    except Exception as e: MODEL_ERROR=str(e)
    STOP.clear(); cleaner=threading.Thread(target=cleanup_loop,daemon=True); cleaner.start()
    LOG.warning('Постоянные данные: %s; временные данные: %s; TTL: %.1f часов',DATA,TEMP,TTL/3600)
    yield
    STOP.set(); cleaner.join(timeout=5); POOL.shutdown(wait=True)

app=FastAPI(title='Liver segmentation — temporary drafts',lifespan=lifespan)

@app.get('/health')
def health(x_client_session: str=Header('')):
    if re.fullmatch(r'[a-f0-9]{32}',x_client_session): touch_session(x_client_session)
    return dict(ok=True,model_ready=ENGINE is not None,device=str(DEVICE),model_error=MODEL_ERROR,
                storage_api=2,draft_ttl_hours=TTL/3600)

@app.get('/studies')
def studies(q: str=''):
    with db() as connection:
        rows=connection.execute('SELECT id,name,revision,updated FROM studies WHERE saved=1 AND name LIKE ? ORDER BY updated DESC',('%'+q+'%',)).fetchall()
    return [dict(r) for r in rows]

@app.post('/studies')
def upload(file: UploadFile=File(...),x_client_session: str=Header('')):
    owner=session_id(x_client_session)
    name=Path(file.filename or 'scan.nii.gz').name
    ext='.nii.gz' if name.lower().endswith('.nii.gz') else Path(name).suffix.lower()
    if ext not in ('.nii','.nii.gz','.zip'): raise HTTPException(400,'Нужен NIfTI или ZIP одной DICOM-серии')
    sid=uuid.uuid4().hex; dest=DRAFTS/sid; dest.mkdir()
    try:
        source=dest/('source'+ext); count=0
        with source.open('wb') as out:
            while chunk:=file.file.read(1024*1024):
                count+=len(chunk)
                if count>MAX_BYTES: raise ValueError('Превышен лимит загрузки')
                out.write(chunk)
        image,meta=load_source(source,dest)
        sitk.WriteImage(image,str(dest/'image.nii.gz'))
        meta.update(spacing=list(image.GetSpacing()),origin=list(image.GetOrigin()),direction=list(image.GetDirection()),size=list(image.GetSize()))
        r=dict(id=sid,name=name,original_name=name,metadata=json.dumps(meta),saved=0,revision=0,
               report='',metrics=None,model=None,owner=owner,created=now_iso(),last_seen=time.time())
        with LOCK: atomic_json(dest/'draft.json',r)
        return dict(id=sid,saved=False)
    except Exception as e:
        shutil.rmtree(dest,ignore_errors=True)
        if isinstance(e,HTTPException): raise
        raise HTTPException(400,str(e))


def run_job(jid,sid,owner):
    try:
        with LOCK:
            JOBS[jid]['status']='running'
            r,path,is_draft=lookup(sid,owner)
            outpath=path if is_draft else result_path(owner,sid)
            outpath.mkdir(parents=True,exist_ok=True)
            if not is_draft:
                atomic_json(outpath/'result.json',dict(id=sid,owner=owner,last_seen=time.time()))
        image=sitk.ReadImage(str(path/'image.nii.gz'))
        def update(value):
            with LOCK: JOBS[jid]['progress']=value
        mask=ENGINE.predict(image,update)
        image_mask=sitk.GetImageFromArray(mask); image_mask.CopyInformation(image)
        sitk.WriteImage(image_mask,str(outpath/f'automatic_{jid}.nii.gz'))
        with LOCK:
            shutil.copyfile(outpath/f'automatic_{jid}.nii.gz',outpath/'automatic.nii.gz')
            atomic_json(outpath/'automatic.json',dict(model=ENGINE.version))
            if is_draft: touch_draft(path,r,True)
            else: atomic_json(outpath/'result.json',dict(id=sid,owner=owner,last_seen=time.time()))
            JOBS[jid].update(status='done',model=ENGINE.version,finished=time.time())
    except Exception as e:
        with LOCK: JOBS[jid].update(status='error',error=str(e),finished=time.time())

@app.post('/studies/{sid}/predict')
def predict(sid: str,x_client_session: str=Header('')):
    owner=session_id(x_client_session)
    with LOCK:
        r,path,draft=lookup(sid,owner)
        if ENGINE is None: raise HTTPException(503,MODEL_ERROR or 'Модель недоступна')
        if active(sid): raise HTTPException(409,'Это исследование уже обрабатывается')
        jid=uuid.uuid4().hex
        JOBS[jid]=dict(study_id=sid,owner=owner,status='queued',progress=0)
        POOL.submit(run_job,jid,sid,owner)
    return dict(job_id=jid)

@app.get('/jobs/{jid}')
def job(jid: str,x_client_session: str=Header('')):
    owner=session_id(x_client_session)
    with LOCK:
        if jid not in JOBS: raise HTTPException(404,'Задача не найдена; сервер мог перезапуститься')
        if JOBS[jid]['owner']!=owner: raise HTTPException(403,'Задача другого сеанса')
        return dict(JOBS[jid])

@app.get('/studies/{sid}/bundle')
def bundle(sid: str,automatic: bool=False,x_client_session: str=Header('')):
    owner=session_id(x_client_session)
    with LOCK:
        r,path,draft=lookup(sid,owner)
        if draft: touch_draft(path,r)
        meta=json.loads(r['metadata'])
        image=sitk.ReadImage(str(path/'image.nii.gz')); volume=sitk.GetArrayFromImage(image).astype(np.float32)
        working=path if draft else result_path(owner,sid)
        maskfile=working/'automatic.nii.gz' if automatic else path/f'mask_{r["revision"]}.nii.gz'
        present=maskfile.exists()
        mask=sitk.GetArrayFromImage(sitk.ReadImage(str(maskfile))).astype(np.uint8) if present else np.zeros(volume.shape,np.uint8)
        model=r.get('model')
        if automatic and (working/'automatic.json').exists(): model=json.loads((working/'automatic.json').read_text(encoding='utf-8')).get('model')
        info=dict(id=sid,name=r['name'],original_name=r.get('original_name') or r['name'],metadata=meta,
                  saved=bool(r['saved']),storage_name=r.get('storage_name'),revision=r['revision'],
                  report=r.get('report') or '',has_mask=present,model=model,
                  metrics=metrics(mask,image.GetSpacing()) if present else None)
        b=io.BytesIO(); np.savez_compressed(b,volume=volume,mask=mask,info=np.array(json.dumps(info)))
        return Response(b.getvalue(),media_type='application/octet-stream')

@app.post('/studies/{sid}/save')
def save(sid: str,mask: UploadFile=File(...),report: str=Form(''),model: str=Form(''),
         has_mask: bool=Form(True),base_revision: int=Form(...),folder_name: str=Form(''),
         x_client_session: str=Header('')):
    owner=session_id(x_client_session)
    with LOCK:
        r,path,draft=lookup(sid,owner)
        if active(sid): raise HTTPException(409,'Дождитесь завершения обработки')
        if base_revision!=r['revision']: raise HTTPException(409,'Версия на сервере изменилась. Откройте исследование заново.')
        first=not bool(r['saved'])
        if first:
            # Legacy unsaved rows are not moved/deleted automatically.
            if not draft: raise HTTPException(409,'Это черновик старой версии. Загрузите исходное КТ заново; старые файлы оставлены без изменений.')
            storage=check_name_available(folder_name)
        else: storage=r.get('storage_name') or sid
        moved=False; written=[]
        oldpath=path; rev=int(r['revision'])+1
        try:
            raw=mask.file.read(MAX_BYTES+1)
            if len(raw)>MAX_BYTES: raise ValueError('Слишком большой файл маски')
            a=np.load(io.BytesIO(raw),allow_pickle=False)
            if not isinstance(a,np.ndarray): raise ValueError('Нужен массив NPY')
            image=sitk.ReadImage(str(path/'image.nii.gz'))
            if a.shape!=tuple(reversed(image.GetSize())) or not np.isin(a,[0,1,2]).all(): raise ValueError('Неверная форма или классы маски')
            m=metrics(a,image.GetSpacing()) if has_mask else None
            if first:
                target=child_path(DATA,storage)
                os.replace(path,target); path=target; moved=True
            if has_mask:
                image_mask=sitk.GetImageFromArray(a.astype(np.uint8)); image_mask.CopyInformation(image)
                tmp=path/f'pending_mask_{rev}.nii.gz'; written.append(tmp)
                sitk.WriteImage(image_mask,str(tmp))
                target=path/f'mask_{rev}.nii.gz'; os.replace(tmp,target); written.append(target)
            snap=path/f'revision_{rev}.json'; written.extend([snap,snap.with_name(snap.name+'.new')])
            atomic_json(snap,dict(revision=rev,report=report,metrics=m,model=model,has_mask=has_mask,created=now_iso()))
            reportfile=path/f'conclusion_{rev}.txt'; written.append(reportfile)
            reportfile.write_text(report,encoding='utf-8')
            # A rerun on saved data is temporary until this explicit save.
            working=result_path(owner,sid)
            if not first and (working/'automatic.nii.gz').exists():
                auto=path/f'automatic_revision_{rev}.nii.gz'; written.append(auto)
                shutil.copyfile(working/'automatic.nii.gz',auto)
            stamp=now_iso()
            with db() as connection:
                if first:
                    connection.execute('INSERT INTO studies (id,name,metadata,saved,revision,report,metrics,model,updated,storage_name,original_name) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                        (sid,storage,r['metadata'],1,rev,report,json.dumps(m),model,stamp,storage,r.get('original_name') or r['name']))
                else:
                    connection.execute('UPDATE studies SET revision=?,report=?,metrics=?,model=?,updated=? WHERE id=?',(rev,report,json.dumps(m),model,stamp,sid))
        except Exception as e:
            for file in written:
                try: file.unlink(missing_ok=True)
                except OSError: LOG.exception('Не удалось убрать незавершённый файл %s',file)
            if moved:
                try: os.replace(path,oldpath)
                except OSError: LOG.exception('Перенос не откатан: будет попытка восстановления при запуске')
            if isinstance(e,HTTPException): raise
            raise HTTPException(400,str(e))
        # SQLite has committed. Cleanup failure here must not be reported as a failed save.
        try:
            if first: (path/'draft.json').unlink(missing_ok=True)
            shutil.rmtree(result_path(owner,sid),ignore_errors=True)
        except OSError: LOG.exception('Сохранено, но не удалось убрать временные файлы %s',sid)
        return dict(id=sid,saved=True,name=storage if first else r['name'],storage_name=storage,revision=rev,metrics=m)


def delete_draft(sid,owner):
    validate_id(sid)
    if saved_row(sid) is not None: raise HTTPException(409,'Сохранённые и старые записи этим методом не удаляются')
    path=DRAFTS/sid
    if not (path/'draft.json').exists(): return
    r=json.loads((path/'draft.json').read_text(encoding='utf-8'))
    if r['owner']!=owner: raise HTTPException(403,'Черновик другого сеанса')
    if active(sid): raise HTTPException(409,'Исследование ещё обрабатывается')
    shutil.rmtree(path)
    for jid in list(JOBS):
        if JOBS[jid]['study_id']==sid and JOBS[jid]['owner']==owner: JOBS.pop(jid)

@app.delete('/studies/{sid}/draft')
def discard_draft(sid: str,x_client_session: str=Header('')):
    owner=session_id(x_client_session)
    with LOCK: delete_draft(sid,owner)
    return dict(deleted=True)

@app.delete('/studies/{sid}/working')
def discard_working(sid: str,x_client_session: str=Header('')):
    owner=session_id(x_client_session); validate_id(sid)
    with LOCK:
        if active(sid,owner): raise HTTPException(409,'Обработка ещё выполняется')
        shutil.rmtree(result_path(owner,sid),ignore_errors=True)
    return dict(deleted=True)

@app.delete('/session/temporary')
def discard_session(x_client_session: str=Header('')):
    owner=session_id(x_client_session)
    with LOCK:
        if any(j['owner']==owner and j['status'] in ('queued','running') for j in JOBS.values()):
            raise HTTPException(409,'Дождитесь завершения задач сеанса')
        for manifest in list(DRAFTS.glob('*/draft.json')):
            r=json.loads(manifest.read_text(encoding='utf-8'))
            if r['owner']==owner: delete_draft(r['id'],owner)
        shutil.rmtree(RESULTS/owner,ignore_errors=True)
    return dict(deleted=True)
