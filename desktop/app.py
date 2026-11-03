"""Desktop client. Server address is configured through SERVER_URL, not in the UI."""
import io, os, sys, json, zipfile, uuid, re
from pathlib import Path
import numpy as np
import cv2
import requests
import math
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF, QRectF, QTranslator, QLibraryInfo
from PyQt5.QtGui import QPainter, QImage, QColor, QPen, QPolygonF
from PyQt5.QtWidgets import (QApplication,QMainWindow,QWidget,QDialog,QVBoxLayout,QHBoxLayout,
    QPushButton,QLabel,QLineEdit,QListWidget,QListWidgetItem,QAbstractItemView,QMessageBox,
    QFileDialog,QSlider,QTextEdit,QComboBox,QSpinBox,QCheckBox,QTableWidget,QTableWidgetItem,
    QSplitter,QInputDialog,QStackedWidget,QScrollBar,QSizePolicy,QGridLayout)

SERVER_URL = os.getenv('SERVER_URL', 'http://127.0.0.1:8000').strip().rstrip('/')

CLIENT_SESSION = uuid.uuid4().hex

LABELS=['Печень без опухоли','Печень с опухолью','Опухоль','Доля опухоли']
KEYS=['healthy','total','tumor','percent']

def calc(mask,spacing):
    k=float(np.prod(spacing))/1000
    h=np.count_nonzero(mask==1)*k; t=np.count_nonzero(mask==2)*k
    return dict(healthy=h,total=h+t,tumor=t,percent=float(t/(h+t)*100) if h+t else None)
def fmt(x): return '—' if x is None else f'{x:.3f}'
def summary(info,mask):
    if not info['has_mask']: return 'Маска ещё не получена.'
    m=calc(mask,info['metadata']['spacing'])
    return '\n'.join(f'{n}: {fmt(m[k])} '+('%' if k=='percent' else 'см³') for n,k in zip(LABELS,KEYS))
def unpack(content):
    with np.load(io.BytesIO(content),allow_pickle=False) as z:
        return z['volume'].copy(),z['mask'].copy(),json.loads(str(z['info'].item()))

def http(base,method,path,**kwargs):
    headers=dict(kwargs.pop('headers',{}))
    headers['X-Client-Session']=CLIENT_SESSION
    kwargs['headers']=headers
    r=requests.request(method,base.rstrip('/')+path,timeout=kwargs.pop('timeout',(8,600)),**kwargs)
    if not r.ok:
        try: msg=r.json().get('detail',r.text)
        except Exception: msg=r.text
        raise RuntimeError(f'HTTP {r.status_code}: {msg}')
    return r

def collect_dicom_series(selected):
    try:
        import pydicom
    except ImportError as e:
        raise RuntimeError('Для DICOM установите зависимость в окружении клиента: python -m pip install pydicom') from e
    tags=['SeriesInstanceUID','SOPInstanceUID','ImagePositionPatient','ImageOrientationPatient','NumberOfFrames']
    try:
        chosen=pydicom.dcmread(str(selected),stop_before_pixels=True,specific_tags=tags)
    except Exception as e:
        raise ValueError('Выбранный файл не распознан как DICOM. Для NIfTI используйте .nii или .nii.gz.') from e
    uid=str(getattr(chosen,'SeriesInstanceUID',''))
    if not uid: raise ValueError('В выбранном файле отсутствует SeriesInstanceUID.')
    if int(getattr(chosen,'NumberOfFrames',1))!=1:
        raise ValueError('Multiframe DICOM пока не поддерживается сервером.')
    matching=[]; seen=set()
    for file in sorted(selected.parent.iterdir()):
        if not file.is_file(): continue
        try:
            d=pydicom.dcmread(str(file),stop_before_pixels=True,specific_tags=tags)
        except Exception: continue
        if str(getattr(d,'SeriesInstanceUID',''))!=uid: continue
        if int(getattr(d,'NumberOfFrames',1))!=1: raise ValueError('В серии есть multiframe DICOM; формат пока не поддерживается.')
        if not hasattr(d,'ImagePositionPatient') or not hasattr(d,'ImageOrientationPatient'): continue
        sop=str(getattr(d,'SOPInstanceUID',''))
        if sop and sop in seen: raise ValueError('В папке есть дубликаты SOPInstanceUID. Удалите дубликаты из папки серии.')
        if sop: seen.add(sop)
        matching.append(file)
    if len(matching)<2:
        raise ValueError('В этой папке не найдена обычная DICOM-серия из двух и более срезов. Файлы серии должны лежать в одной папке.')
    return matching


class Task(QThread):
    result=pyqtSignal(object); error=pyqtSignal(str)
    def __init__(self,fn): super().__init__(); self.fn=fn
    def run(self):
        try: self.result.emit(self.fn())
        except Exception as e: self.error.emit(str(e))

class Orientation:
    """Only lossless in-plane flips/transposition. Never applied to model input."""
    def __init__(self,meta):
        d=np.array(meta['direction']).reshape(3,3)
        target=np.zeros((3,3)); dominant=np.argmax(abs(d),axis=0)
        for i,j in enumerate(dominant): target[j,i]=np.sign(d[j,i])
        self.known=len(set(dominant))==3 and np.allclose(d,target,atol=1e-4) and dominant[2]==2
        self.transpose=False; self.fx=False; self.fy=False
        sp=meta['spacing']; self.px=sp[0]; self.py=sp[1]
        if self.known:
            self.transpose=dominant[0]==1
            ix=1 if self.transpose else 0; iy=0 if self.transpose else 1
            self.fx=d[0,ix]<0; self.fy=d[1,iy]<0
            self.px=sp[ix]; self.py=sp[iy]
    def forward(self,a):
        if self.transpose: a=a.T
        if self.fx: a=a[:,::-1]
        if self.fy: a=a[::-1,:]
        return np.ascontiguousarray(a)
    def inverse(self,a):
        if self.fy: a=a[::-1,:]
        if self.fx: a=a[:,::-1]
        if self.transpose: a=a.T
        return np.ascontiguousarray(a)

class Canvas(QWidget):
    navigation=pyqtSignal(object)
    def __init__(self,panel,overlay):
        super().__init__(); self.panel=panel; self.overlay=overlay; self.setMinimumSize(230,230)
        self.zoom=1.; self.pan=QPointF(); self.last=None; self.active=False; self.point=None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.scroller=None
    def mapping(self):
        p=self.panel; a=p.orientation.forward(p.volume[p.index]); h,w=a.shape
        px=p.orientation.px; py=p.orientation.py
        fit=min(max(1,self.width()-20)/(w*px),max(1,self.height()-40)/(h*py))
        sx=fit*self.zoom*px; sy=fit*self.zoom*py
        return QPointF((self.width()-w*sx)/2,(self.height()-h*sy)/2)+self.pan,sx,sy,w,h
    def image_pos(self,pos):
        o,sx,sy,w,h=self.mapping()
        return QPointF((pos.x()-o.x())/sx,(pos.y()-o.y())/sy)
    def paintEvent(self,event):
        p=QPainter(self); p.fillRect(self.rect(),Qt.black)
        panel=self.panel
        a=panel.orientation.forward(panel.volume[panel.index])
        gray=(np.clip((a+150)/400,0,1)*255).astype(np.uint8)
        rgb=np.repeat(gray[:,:,None],3,axis=2)
        if self.overlay and panel.visible_mask and panel.info['has_mask']:
            m=panel.orientation.forward(panel.mask[panel.index])
            for cls,color in [(1,[0,255,0]),(2,[255,0,0])]:
                sel=m==cls; rgb[sel]=(.55*rgb[sel]+.45*np.array(color)).astype(np.uint8)
        rgb=np.ascontiguousarray(rgb); h,w=gray.shape
        img=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy()
        o,sx,sy,_,_=self.mapping(); p.drawImage(QRectF(o.x(),o.y(),w*sx,h*sy),img)
        if panel.editor and panel.editor.points is not None:
            pts=panel.editor.points
            poly=QPolygonF([QPointF(o.x()+x*sx,o.y()+y*sy) for x,y in pts])
            p.setPen(QPen(QColor('yellow'),2))
            if panel.editor.polygon_closed:
                p.drawPolygon(poly)
            else:
                p.drawPolyline(poly)
            for pt in poly: p.drawEllipse(pt,4,4)
        p.setPen(Qt.white)
        if panel.orientation.known:
            p.drawText(self.width()//2,16,'A'); p.drawText(self.width()//2,self.height()-5,'P')
            p.drawText(4,self.height()//2,'R'); p.drawText(self.width()-15,self.height()//2,'L')
        else: p.drawText(8,18,'Исходная плоскость: ориентация НЕ стандартизована')
        p.end()
    def mousePressEvent(self,e):
        self.setFocus(Qt.MouseFocusReason)
        if e.button()==Qt.RightButton:
            self.last=e.globalPos(); self.setCursor(Qt.ClosedHandCursor); e.accept(); return
        editor=self.panel.editor
        if e.button()!=Qt.LeftButton or not editor: return
        position=self.image_pos(e.pos()); x,y=position.x(),position.y()
        h,w=editor.display_mask().shape
        if not (0<=x<w and 0<=y<h): return
        if editor.tool.currentIndex()==1:
            if not editor.polygon_closed:
                editor.add_vertex(x,y)
            elif editor.points is not None:
                origin,sx,sy,_,_=self.mapping()
                screen=np.array([[origin.x()+p[0]*sx,origin.y()+p[1]*sy] for p in editor.points])
                distance=np.linalg.norm(screen-np.array([e.pos().x(),e.pos().y()]),axis=1)
                index=int(distance.argmin())
                if distance[index]<=14:
                    self.point=index; editor.begin_gesture()
        else:
            editor.clear_selection(); editor.begin_gesture()
            self.active=True; self.last_point=(int(x),int(y)); editor.stroke(self.last_point,self.last_point)
        e.accept()
    def mouseMoveEvent(self,e):
        if e.buttons() & Qt.RightButton:
            position=e.globalPos()
            if self.last is not None:
                delta=position-self.last; self.scroll_by(delta.x(),delta.y())
            self.last=position; e.accept(); return
        editor=self.panel.editor
        if not editor or not (e.buttons() & Qt.LeftButton): return
        position=self.image_pos(e.pos()); h,w=editor.display_mask().shape
        x=float(np.clip(position.x(),0,w-1)); y=float(np.clip(position.y(),0,h-1))
        if self.point is not None and editor.points is not None and editor.polygon_closed:
            editor.points[self.point]=[x,y]; self.update()
        elif self.active:
            point=(int(x),int(y)); editor.stroke(self.last_point,point); self.last_point=point
        e.accept()
    def mouseReleaseEvent(self,e):
        if e.button()==Qt.RightButton:
            self.last=None; self.unsetCursor(); e.accept(); return
        if e.button()==Qt.LeftButton:
            if self.panel.editor: self.panel.editor.end_gesture()
            self.active=False; self.point=None; e.accept(); return
        super().mouseReleaseEvent(e)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scroller:
            QTimer.singleShot(0, self.scroller.update_bars)

    def scroll_by(self, dx, dy):
        self.pan += QPointF(dx, dy)
        self.navigation.emit((self.zoom, QPointF(self.pan)))

    def keyPressEvent(self, event):
        step = 40
        offsets = {
            Qt.Key_Left: (step, 0), Qt.Key_Right: (-step, 0),
            Qt.Key_Up: (0, step), Qt.Key_Down: (0, -step),
        }
        if event.key() in offsets:
            self.scroll_by(*offsets[event.key()]); event.accept()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        delta = event.pixelDelta()
        if not delta.isNull():
            dx, dy = delta.x(), delta.y()
        else:
            angle = event.angleDelta()
            dx, dy = angle.x()/120*60, angle.y()/120*60
        if event.modifiers() & Qt.ShiftModifier:
            dx, dy = dy, dx
        self.scroll_by(dx, dy)
        event.accept()


class ScrollCanvas(QWidget):
    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas
        canvas.scroller = self
        self.horizontal = QScrollBar(Qt.Horizontal)
        self.vertical = QScrollBar(Qt.Vertical)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0); grid.setSpacing(0)
        grid.addWidget(canvas, 0, 0)
        grid.addWidget(self.vertical, 0, 1)
        grid.addWidget(self.horizontal, 1, 0)
        grid.setRowStretch(0, 1); grid.setColumnStretch(0, 1)
        for bar in (self.horizontal, self.vertical):
            policy=bar.sizePolicy(); policy.setRetainSizeWhenHidden(True); bar.setSizePolicy(policy)
            bar.setSingleStep(40)
            bar.hide()
        self.horizontal.valueChanged.connect(self.scrolled)
        self.vertical.valueChanged.connect(self.scrolled)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def limits(self):
        c=self.canvas
        _,sx,sy,w,h=c.mapping()
        return max(0,math.ceil(w*sx-c.width())),max(0,math.ceil(h*sy-c.height()))

    def update_bars(self):
        c=self.canvas
        hx,vy=self.limits()
        c.pan=QPointF(float(np.clip(c.pan.x(),-hx/2,hx/2)),float(np.clip(c.pan.y(),-vy/2,vy/2)))
        for bar,limit,value,page in [
            (self.horizontal,hx,hx/2-c.pan.x(),c.width()),
            (self.vertical,vy,vy/2-c.pan.y(),c.height()),
        ]:
            bar.blockSignals(True)
            bar.setRange(0,limit); bar.setPageStep(page); bar.setValue(round(value))
            bar.setVisible(limit>0)
            bar.blockSignals(False)
        c.update()

    def scrolled(self):
        hx,vy=self.limits()
        c=self.canvas
        c.pan=QPointF(hx/2-self.horizontal.value(),vy/2-self.vertical.value())
        c.navigation.emit((c.zoom,QPointF(c.pan)))

class SlicePanel(QWidget):
    def __init__(self,volume,mask,info,single=False):
        super().__init__()
        self.volume=volume; self.mask=mask; self.info=info
        self.orientation=Orientation(info['metadata']); self.index=len(volume)//2
        self.visible_mask=True; self.editor=None
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        layout=QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(5)
        self.title=QLabel(info['name']); layout.addWidget(self.title)
        row=QHBoxLayout(); row.setSpacing(8); self.views=[]
        for overlay in ([True] if single else [False,True]):
            column=QVBoxLayout(); column.setSpacing(3)
            caption=QLabel('КТ с маской' if overlay else 'Исходное КТ')
            caption.setAlignment(Qt.AlignCenter); column.addWidget(caption)
            view=Canvas(self,overlay)
            view.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
            self.views.append(view); column.addWidget(ScrollCanvas(view),1)
            view.navigation.connect(self.sync)
            row.addLayout(column,1)
        layout.addLayout(row,1)
        slices=QHBoxLayout()
        self.slice_label=QLabel(); slices.addWidget(self.slice_label)
        self.slider=QSlider(Qt.Horizontal); self.slider.setRange(0,len(volume)-1); self.slider.setValue(self.index)
        self.slider.valueChanged.connect(self.change_slice); slices.addWidget(self.slider,1)
        layout.addLayout(slices)
        nav=QHBoxLayout()
        self.zoom=QSlider(Qt.Horizontal); self.zoom.setRange(25,600); self.zoom.setValue(100)
        self.zoom.setSingleStep(10); self.zoom.setPageStep(10)
        self.zoom_label=QLabel('100%'); self.zoom_label.setMinimumWidth(48)
        minus=QPushButton('−'); plus=QPushButton('+'); fit=QPushButton('Вписать')
        minus.setFixedWidth(34); plus.setFixedWidth(34)
        minus.clicked.connect(lambda:self.zoom.setValue(self.zoom.value()-10))
        plus.clicked.connect(lambda:self.zoom.setValue(self.zoom.value()+10))
        fit.clicked.connect(self.fit_image)
        nav.addWidget(QLabel('Масштаб')); nav.addWidget(minus); nav.addWidget(self.zoom_label)
        nav.addWidget(plus); nav.addWidget(self.zoom,1); nav.addWidget(fit)
        self.zoom_label.setToolTip('100% — весь срез вписан в окно')
        self.zoom.valueChanged.connect(self.set_zoom); layout.addLayout(nav)
        self.numbers=QLabel(); self.numbers.setWordWrap(True)
        self.numbers.setTextInteractionFlags(Qt.TextSelectableByMouse); layout.addWidget(self.numbers)
        self.refresh()
    def sync(self,state):
        zoom,pan=state
        for view in self.views:
            view.zoom=zoom; view.pan=QPointF(pan)
            view.scroller.update_bars()
    def set_zoom(self,value):
        self.zoom_label.setText(f'{value}%')
        self.sync((value/100,self.views[0].pan))

    def fit_image(self):
        self.zoom.setValue(100)
        self.sync((1.,QPointF()))
    def change_slice(self,value):
        if self.editor: self.editor.clear_selection()
        self.index=value; self.refresh(False)
    def refresh(self,numbers=True):
        self.slice_label.setText(f'Срез {self.index+1} / {len(self.volume)}')
        if numbers:
            self.numbers.setText(summary(self.info,self.mask).replace('\n','    |    '))
        for view in self.views:
            view.update()

class Editor(QDialog):
    def __init__(self,parent,panel):
        super().__init__(parent)
        self.setWindowTitle('Редактор маски')
        self.setWindowFlag(Qt.WindowContextHelpButtonHint,False)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint,True)
        self.resize(1100,850)
        self.points=None; self.polygon_closed=False
        self.undo_stack=[]; self.redo_stack=[]; self.gesture_before=None
        self.panel=SlicePanel(panel.volume,panel.mask.copy(),dict(panel.info),True)
        self.panel.editor=self
        layout=QVBoxLayout(self); tools=QGridLayout()
        self.tool=QComboBox(); self.tool.addItems(['Кисть','Замкнутая область'])
        self.cls=QComboBox(); self.cls.addItems(['Фон','Печень','Опухоль']); self.cls.setCurrentIndex(1)
        self.radius=QSpinBox(); self.radius.setRange(1,100); self.radius.setValue(8)
        for col,(label,widget) in enumerate([('Инструмент',self.tool),('Класс',self.cls),('Радиус кисти',self.radius)]):
            tools.addWidget(QLabel(label),0,col*2); tools.addWidget(widget,0,col*2+1)
        layout.addLayout(tools)
        self.overwrite=QCheckBox('Разрешить замену другого класса (включить также для стирания фоном)')
        layout.addWidget(self.overwrite)
        self.hint=QLabel(); self.hint.setWordWrap(True); layout.addWidget(self.hint)
        layout.addWidget(self.panel,1)
        buttons=QGridLayout(); self.action_buttons={}
        actions=[('Скрыть / показать маску',self.toggle),('Завершить построение',self.finish_polygon),
                 ('Залить область',self.fill_polygon),('Сбросить область',self.reset_polygon),
                 ('Отмена действия',self.undo),('Повтор действия',self.redo),
                 ('Применить изменения',self.accept),('Закрыть без применения',self.reject)]
        for i,(text,fn) in enumerate(actions):
            button=QPushButton(text); button.clicked.connect(fn)
            buttons.addWidget(button,i//4,i%4); self.action_buttons[text]=button
        layout.addLayout(buttons)
        self.tool.currentIndexChanged.connect(self.tool_changed)
        self.update_hint()

    def update_hint(self):
        if self.tool.currentIndex()==0:
            text='Кисть: рисование ЛКМ. Для стирания выберите «Фон» и разрешите замену класса. ПКМ — перемещение изображения.'
        elif self.polygon_closed:
            text='Область замкнута. Перетаскивайте кружки зажатой ЛКМ, затем нажмите «Залить область». Новые точки не добавляются. ПКМ — перемещение.'
        else:
            count=0 if self.points is None else len(self.points)
            text=f'Построение области: добавляйте вершины щелчками ЛКМ (точек: {count}). Затем «Завершить построение». Отмена убирает последнюю точку; повтор возвращает её.'
        self.hint.setText(text)
        polygon_mode=self.tool.currentIndex()==1
        self.action_buttons['Завершить построение'].setEnabled(polygon_mode and not self.polygon_closed and self.points is not None and len(self.points)>=3)
        self.action_buttons['Залить область'].setEnabled(polygon_mode and self.polygon_closed)
        self.action_buttons['Сбросить область'].setEnabled(self.points is not None)
        self.action_buttons['Отмена действия'].setEnabled(bool(self.undo_stack))
        self.action_buttons['Повтор действия'].setEnabled(bool(self.redo_stack))

    def tool_changed(self,*args):
        self.clear_selection()

    def clear_selection(self,*args):
        self.points=None; self.polygon_closed=False; self.gesture_before=None
        for view in self.panel.views:
            view.point=None; view.active=False; view.update()
        if hasattr(self,'hint') and hasattr(self,'action_buttons'):
            self.update_hint()

    def reset_polygon(self):
        if self.points is None: return
        before=self.capture(); self.clear_selection(); self.commit(before)

    def display_mask(self):
        return self.panel.orientation.forward(self.panel.mask[self.panel.index])

    def write(self,a):
        self.panel.mask[self.panel.index]=self.panel.orientation.inverse(a)
        self.panel.info['has_mask']=True
        self.panel.refresh(False)

    def capture(self):
        return dict(z=self.panel.index,mask=self.panel.mask[self.panel.index].copy(),
                    has_mask=self.panel.info['has_mask'],
                    points=None if self.points is None else self.points.copy(),
                    closed=self.polygon_closed,tool=self.tool.currentIndex())

    def commit(self,before):
        after=self.capture()
        same_points=(before['points'] is None and after['points'] is None) or (
            before['points'] is not None and after['points'] is not None and
            np.array_equal(before['points'],after['points']))
        if np.array_equal(before['mask'],after['mask']) and same_points and before['has_mask']==after['has_mask'] and before['closed']==after['closed']:
            self.update_hint(); return
        self.undo_stack.append((before,after)); self.undo_stack=self.undo_stack[-50:]
        self.redo_stack.clear(); self.update_hint()

    def begin_gesture(self):
        self.gesture_before=self.capture()

    def end_gesture(self):
        if self.gesture_before is not None:
            before=self.gesture_before; self.gesture_before=None
            self.commit(before)

    def add_vertex(self,x,y):
        if self.polygon_closed: return
        before=self.capture()
        if self.points is None: self.points=np.empty((0,2),np.float32)
        self.points=np.vstack([self.points,np.array([[x,y]],np.float32)])
        self.panel.refresh(False); self.commit(before)

    def stroke(self,start,end):
        a=self.display_mask(); region=np.zeros(a.shape,np.uint8)
        cv2.line(region,start,end,1,2*self.radius.value())
        cv2.circle(region,end,self.radius.value(),1,-1)
        cls=self.cls.currentIndex(); allowed=(a==0)|(a==cls)
        if self.overwrite.isChecked(): allowed[:]=True
        a[(region>0)&allowed]=cls; self.write(a)

    def valid_polygon(self):
        if self.points is None or len(self.points)<3:
            QMessageBox.information(self,'Область','Добавьте не менее трёх точек.'); return False
        from shapely.geometry import Polygon
        polygon=Polygon(self.points)
        if not polygon.is_valid or polygon.area<1:
            QMessageBox.warning(self,'Область','Граница самопересекается или имеет нулевую площадь. Исправьте точки или отмените последнее действие.'); return False
        return True

    def finish_polygon(self):
        if self.polygon_closed or not self.valid_polygon(): return
        before=self.capture(); self.polygon_closed=True
        self.panel.refresh(False); self.commit(before)

    def region(self):
        a=np.zeros(self.display_mask().shape,np.uint8)
        cv2.fillPoly(a,[np.rint(self.points).astype(np.int32)],1)
        return a.astype(bool)

    def fill_polygon(self):
        if not self.polygon_closed or not self.valid_polygon(): return False
        before=self.capture(); a=self.display_mask(); cls=self.cls.currentIndex()
        allowed=(a==0)|(a==cls)
        if self.overwrite.isChecked(): allowed[:]=True
        selected=self.region() & allowed
        if not selected.any():
            QMessageBox.information(self,'Заливка','Нет доступных пикселей. Для замены существующего класса включите соответствующий флажок.'); return False
        a[selected]=cls
        self.write(a); self.clear_selection(); self.panel.refresh()
        self.commit(before)
        return True

    def restore(self,state):
        z=state['z']; self.panel.mask[z]=state['mask'].copy()
        self.panel.info['has_mask']=state['has_mask']
        self.panel.slider.blockSignals(True); self.panel.slider.setValue(z); self.panel.slider.blockSignals(False)
        self.panel.index=z
        self.tool.blockSignals(True); self.tool.setCurrentIndex(state['tool']); self.tool.blockSignals(False)
        self.points=None if state['points'] is None else state['points'].copy()
        self.polygon_closed=state['closed']; self.gesture_before=None
        for view in self.panel.views: view.point=None; view.active=False
        self.panel.refresh(); self.update_hint()

    def undo(self):
        self.end_gesture()
        if not self.undo_stack: return
        before,after=self.undo_stack.pop(); self.redo_stack.append((before,after))
        self.restore(before)

    def redo(self):
        if not self.redo_stack: return
        before,after=self.redo_stack.pop(); self.undo_stack.append((before,after))
        self.restore(after)

    def toggle(self):
        self.panel.visible_mask=not self.panel.visible_mask; self.panel.refresh(False)

    def accept(self):
        self.end_gesture()
        if self.points is not None:
            message=QMessageBox(self); message.setWindowTitle('Незавершённая область')
            message.setText('Залить текущую область перед применением изменений редактора?')
            message.setStandardButtons(QMessageBox.Save|QMessageBox.Discard|QMessageBox.Cancel)
            message.button(QMessageBox.Save).setText('Залить и применить')
            message.button(QMessageBox.Discard).setText('Без этой области')
            message.button(QMessageBox.Cancel).setText('Отмена')
            answer=message.exec_()
            if answer==QMessageBox.Cancel: return
            if answer==QMessageBox.Save:
                if not self.polygon_closed:
                    self.finish_polygon()
                    if not self.polygon_closed: return
                if not self.fill_polygon(): return
        self.clear_selection()
        super().accept()

class Comparison(QDialog):
    def __init__(self,parent,bundles):
        super().__init__(parent); self.setWindowFlag(Qt.WindowContextHelpButtonHint,False); self.setWindowTitle('Было → стало (порядок выбран пользователем)'); self.resize(1450,900)
        layout=QVBoxLayout(self); row=QHBoxLayout(); vals=[]
        for v,m,info in bundles:
            row.addWidget(SlicePanel(v,m,info)); vals.append(calc(m,info['metadata']['spacing']) if info['has_mask'] else None)
        layout.addWidget(QLabel('Слева — было, справа — стало. Срезы независимы. Принадлежность одному пациенту не проверяется.'))
        layout.addLayout(row,1); table=QTableWidget(4,5); table.setHorizontalHeaderLabels(['Показатель','Было','Стало','Абсолютное Δ','Относительное Δ, %'])
        for i,(name,key) in enumerate(zip(LABELS,KEYS)):
            a=vals[0][key] if vals[0] else None; b=vals[1][key] if vals[1] else None
            delta=b-a if a is not None and b is not None else None
            relative=100*delta/a if delta is not None and a!=0 else None
            unit='п. п.' if key=='percent' else 'см³'
            for j,text in enumerate([name,fmt(a),fmt(b),fmt(delta)+' '+unit,fmt(relative)]): table.setItem(i,j,QTableWidgetItem(text))
        table.resizeColumnsToContents(); table.setMaximumHeight(175); layout.addWidget(table)

class ReportDialog(QDialog):
    def __init__(self,owner):
        super().__init__(owner); self.setWindowFlag(Qt.WindowContextHelpButtonHint,False); self.owner=owner; self.generated=False
        self.setWindowTitle('Заключение'); self.resize(850,650)
        layout=QVBoxLayout(self)
        self.note=QLabel(); self.note.setWordWrap(True); layout.addWidget(self.note)
        self.text=QTextEdit(); layout.addWidget(self.text,1)
        existing=owner.report.toPlainText()
        if existing:
            self.text.setPlainText(existing)
        else:
            self.text.setPlainText(owner.report_template()); self.generated=True
        self.update_note()
        row=QHBoxLayout()
        for title,fn in [('Обновить шаблон',self.regenerate),('Сохранить TXT',self.export_txt),('Применить',self.accept),('Закрыть',self.reject)]:
            button=QPushButton(title); button.clicked.connect(fn); row.addWidget(button)
        layout.addLayout(row)

    def update_note(self):
        stale=self.owner.report_stale and not self.generated
        self.note.setText('Маска менялась: проверьте численные данные в тексте. Обновление шаблона заменит текст целиком.' if stale else 'Применение сохраняет текст в открытом исследовании. Для записи на сервер нажмите «Сохранить исследование».')

    def regenerate(self):
        if self.text.toPlainText().strip():
            if QMessageBox.question(self,'Обновление','Заменить весь текст новым шаблоном? Ручные комментарии будут удалены.',QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes: return
        self.text.setPlainText(self.owner.report_template()); self.generated=True; self.update_note()

    def apply_text(self):
        value=self.text.toPlainText()
        if value!=self.owner.report.toPlainText(): self.owner.report.setPlainText(value)
        if self.generated: self.owner.report_stale=False

    def export_txt(self):
        if self.owner.report_stale and not self.generated:
            if QMessageBox.question(self,'Проверка текста','Маска менялась. Сохранить этот текст в TXT без автоматического обновления чисел?',QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes: return
        name=Path(self.owner.panel.info['name']).name
        if name.lower().endswith('.nii.gz'): name=name[:-7]
        else: name=Path(name).stem
        path,_=QFileDialog.getSaveFileName(self,'Сохранить заключение',name+'_заключение.txt','Текст (*.txt)')
        if not path: return
        if not path.lower().endswith('.txt'): path+='.txt'
        try:
            Path(path).write_text(self.text.toPlainText(),encoding='utf-8')
        except Exception as e:
            QMessageBox.critical(self,'Ошибка сохранения',str(e)); return
        self.apply_text()
        QMessageBox.information(self,'Готово','TXT сохранён. Текст также применён к открытому исследованию; серверное сохранение выполняется отдельно.')

    def accept(self):
        self.apply_text(); super().accept()

    def reject(self):
        if self.text.toPlainText()!=self.owner.report.toPlainText():
            answer=QMessageBox.question(self,'Изменения текста','Применить изменения к открытому исследованию?',QMessageBox.Save|QMessageBox.Discard|QMessageBox.Cancel,QMessageBox.Save)
            if answer==QMessageBox.Cancel: return
            if answer==QMessageBox.Save: self.apply_text()
        super().reject()


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('КТ печени — исследовательский прототип')
        self.resize(1450,950)
        self.panel=None; self.dirty=False; self.report_stale=False
        self.busy=False; self.tasks=[]; self.job_id=None; self.polling=False
        self.server_url=SERVER_URL
        self._close_authorized=False; self._closing=False; self._session_has_uploads=False
        central=QWidget(); self.setCentralWidget(central)
        layout=QVBoxLayout(central); layout.setContentsMargins(10,8,10,8); layout.setSpacing(6)
        top=QHBoxLayout()
        self.warning=QLabel('Автоматическая разметка и объёмы требуют проверки специалистом.')
        self.warning.setStyleSheet('color: #935000;'); self.warning.hide()
        top.addWidget(self.warning); top.addStretch()
        self.health_label=QLabel('● Не подключён'); top.addWidget(self.health_label)
        layout.addLayout(top)
        self.pages=QStackedWidget(); layout.addWidget(self.pages,1)
        self.home=QWidget(); home_layout=QHBoxLayout(self.home)
        home_layout.setContentsMargins(0,0,0,0); home_layout.setSpacing(12)
        history_column=QWidget(); history_column.setMinimumWidth(240)
        history_column.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Expanding)
        column=QVBoxLayout(history_column); column.setContentsMargins(0,0,0,0)
        column.addWidget(QLabel('История исследований'))
        self.search=QLineEdit(); self.search.setPlaceholderText('Поиск по названию — Enter')
        column.addWidget(self.search)
        self.history=QListWidget(); self.history.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.history.itemDoubleClicked.connect(lambda item:self.open_selected())
        self.history.setTextElideMode(Qt.ElideRight)
        column.addWidget(self.history,1)
        hint=QLabel('Для сравнения выберите два исследования с Ctrl.'); hint.setWordWrap(True)
        column.addWidget(hint)
        for text,fn in [('Загрузить КТ с компьютера',self.new),('Загрузить КТ из истории',self.open_selected),('Сравнение динамики',self.compare),('Обновить историю',self.refresh_history)]:
            button=QPushButton(text); button.clicked.connect(fn); column.addWidget(button)
        home_layout.addWidget(history_column,1)
        empty_space=QWidget(); empty_space.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Expanding)
        home_layout.addWidget(empty_space,3)
        self.pages.addWidget(self.home)
        self.study_page=QWidget(); self.right_layout=QVBoxLayout(self.study_page)
        self.right_layout.setContentsMargins(0,0,0,0); self.right_layout.setSpacing(5)
        actions=QHBoxLayout(); self.study_buttons=[]
        for text,fn in [('К истории',self.go_home),('Обработать изображение',self.predict),('Редактировать разметку',self.edit),('Заключение',self.generate),('Информация',self.show_metadata),('Сохранить исследование',self.save)]:
            b=QPushButton(text); b.clicked.connect(fn); actions.addWidget(b); self.study_buttons.append(b)
        self.right_layout.addLayout(actions)
        self.pages.addWidget(self.study_page); self.pages.setCurrentWidget(self.home)
        self.report=QTextEdit(self); self.report.hide()
        self.report.textChanged.connect(self.report_changed)
        self.status=self.statusBar(); self.status.showMessage('Готово')
        self.timer=QTimer(self); self.timer.timeout.connect(self.poll); self.timer.start(1500)
        self.health_timer=QTimer(self); self.health_timer.timeout.connect(self.check_health); self.health_timer.start(6000)
        self.search.returnPressed.connect(self.refresh_history)
        QTimer.singleShot(0,self.check_health)
        QTimer.singleShot(200,self.refresh_history)
    def base(self):
        return self.server_url

    def go_home(self):
        def leave():
            self.pages.setCurrentWidget(self.home); self.warning.hide()
            if self.panel:
                self.right_layout.removeWidget(self.panel); self.panel.deleteLater(); self.panel=None
            self.report.blockSignals(True); self.report.clear(); self.report.blockSignals(False)
            self.dirty=False; self.report_stale=False
            self.refresh_history()
        self.guard(leave)

    def show_metadata(self):
        if not self.panel: return
        dialog=QDialog(self); dialog.setWindowFlag(Qt.WindowContextHelpButtonHint,False); dialog.setWindowTitle('Информация об исследовании'); dialog.resize(650,500)
        layout=QVBoxLayout(dialog); text=QTextEdit(); text.setReadOnly(True)
        text.setPlainText(json.dumps(self.panel.info['metadata'],ensure_ascii=False,indent=2))
        layout.addWidget(text); close=QPushButton('Закрыть'); close.clicked.connect(dialog.accept); layout.addWidget(close)
        dialog.exec_()
    def run(self,fn,done,label='Запрос...',block_ui=True):
        if self.busy: return
        self.busy=True
        if block_ui: self.centralWidget().setEnabled(False)
        if label: self.status.showMessage(label)
        task=Task(fn); self.tasks.append(task)
        def complete(value):
            self.busy=False
            if block_ui: self.centralWidget().setEnabled(True)
            if label: self.status.showMessage('Готово')
            try: done(value)
            except Exception as e: QMessageBox.critical(self,'Ошибка интерфейса',str(e))
        def failed(msg):
            self.busy=False
            if block_ui: self.centralWidget().setEnabled(True)
            self.status.showMessage('Ошибка запроса')
            QMessageBox.critical(self,'Ошибка запроса',msg)
        task.result.connect(complete); task.error.connect(failed)
        def cleanup():
            self.tasks.remove(task); task.deleteLater()
        task.finished.connect(cleanup); task.start()
    def check_health(self):
        if self._closing: return
        if self.busy or getattr(self,'checking',False): return
        self.checking=True; base=self.base()
        task=Task(lambda:http(base,'GET','/health',timeout=(2,3)).json()); self.tasks.append(task)
        def ok(value):
            self.health_label.setText('● Подключён'); self.health_label.setStyleSheet('color: #247a36;')
            self.health_label.setToolTip(('Модель готова' if value.get('model_ready') else 'Модель не загружена: '+str(value.get('model_error') or '')) if value.get('storage_api')==2 else 'Нужен обновлённый server.py: старый сервер не поддерживает временные исследования.')
        def error(message):
            self.health_label.setText('● Не подключён'); self.health_label.setStyleSheet('color: #a52828;')
            self.health_label.setToolTip(message)
        task.result.connect(ok); task.error.connect(error)
        def finish():
            self.checking=False; self.tasks.remove(task); task.deleteLater()
        task.finished.connect(finish); task.start()
    def refresh_history(self):
        if self._closing: return
        base=self.base(); q=self.search.text()
        def done(rows):
            self.history.clear()
            for r in rows:
                item=QListWidgetItem(f'{r["name"]}\n{r["updated"][:19]} | v{r["revision"]}'); item.setData(Qt.UserRole,r['id']); self.history.addItem(item)
        self.run(lambda:http(base,'GET','/studies',params={'q':q}).json(),done)
    def report_changed(self):
        if self.panel: self.dirty=True
    def guard(self,continuation):
        if self.busy: return
        if self.job_id:
            QMessageBox.information(self,'Обработка','Дождитесь завершения обработки перед сменой исследования.'); return
        if not self.dirty:
            if self.is_draft(): self.discard_current(continuation)
            else: continuation()
            return
        message=QMessageBox(self); message.setWindowTitle('Изменения')
        message.setIcon(QMessageBox.Question)
        message.setText('Сохранить исследование на сервере?' if self.is_draft() else 'Сохранить изменения на сервере?')
        message.setInformativeText('При отказе временная загрузка будет удалена.' if self.is_draft() else 'Сохранённые ранее версии останутся без изменений.')
        message.setStandardButtons(QMessageBox.Save|QMessageBox.Discard|QMessageBox.Cancel)
        message.button(QMessageBox.Save).setText('Сохранить')
        message.button(QMessageBox.Discard).setText('Не сохранять')
        message.button(QMessageBox.Cancel).setText('Отмена')
        message.setDefaultButton(QMessageBox.Save); message.setEscapeButton(QMessageBox.Cancel)
        answer=message.exec_()
        if answer==QMessageBox.Save: self.save(after=continuation)
        elif answer==QMessageBox.Discard: self.discard_current(continuation)

    def is_draft(self):
        return self.panel is not None and not bool(self.panel.info.get('saved',int(self.panel.info.get('revision',0))>0))

    def discard_current(self,continuation):
        if not self.panel:
            continuation(); return
        base=self.base(); sid=self.panel.info['id']; draft=self.is_draft()
        route=f'/studies/{sid}/draft' if draft else f'/studies/{sid}/working'
        def request():
            try:
                http(base,'DELETE',route,timeout=(3,8)); return None
            except Exception as e: return str(e)
        def done(error):
            if error:
                answer=QMessageBox.question(self,'Очистка не подтверждена',
                    'Не удалось подтвердить удаление временных файлов:\n'+error+
                    '\n\nПродолжить? Просроченные черновики удаляются сервером автоматически; старые серверные версии не поддерживают эту очистку.',
                    QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
                if answer!=QMessageBox.Yes: return
            self.dirty=False; continuation()
        self.run(request,done,'Удаление временных данных...')
    def install(self,bundle):
        volume,mask,info=bundle
        if self.panel:
            self.right_layout.removeWidget(self.panel); self.panel.deleteLater()
        self.panel=SlicePanel(volume,mask,info)
        self.right_layout.addWidget(self.panel,1)
        self.report.blockSignals(True); self.report.setPlainText(info.get('report') or ''); self.report.blockSignals(False)
        self.dirty=False; self.report_stale=False
        self.pages.setCurrentWidget(self.study_page); self.warning.show()
        QTimer.singleShot(0,self.panel.fit_image)
        if not self.panel.orientation.known:
            QMessageBox.warning(self,'Ориентация','Автоматически ориентируются только осевые серии без косого наклона. Показана исходная плоскость без меток A/P/R/L.')
    def new(self): self.guard(self.choose_new)
    def choose_new(self):
        path,_=QFileDialog.getOpenFileName(
            self,'Выберите NIfTI или любой DICOM-файл нужной серии','',
            'КТ-изображения (*.nii *.nii.gz *.dcm *.DCM);;Все файлы (*)')
        if not path: return
        self._session_has_uploads=True
        base=self.base(); source=Path(path)
        nifti=source.name.lower().endswith(('.nii','.nii.gz'))
        def upload_and_open():
            sid=None
            try:
                if nifti:
                    with source.open('rb') as file:
                        sid=http(base,'POST','/studies',files={'file':(source.name,file)}).json()['id']
                else:
                    files=collect_dicom_series(source)
                    with io.BytesIO() as buffer:
                        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
                            for index,file in enumerate(files): archive.write(file,f'{index:06d}.dcm')
                        buffer.seek(0)
                        sid=http(base,'POST','/studies',files={'file':(source.stem+'_series.zip',buffer)}).json()['id']
                return unpack(http(base,'GET',f'/studies/{sid}/bundle').content)
            except Exception:
                if sid:
                    try: http(base,'DELETE',f'/studies/{sid}/draft',timeout=(3,5))
                    except Exception: pass
                raise
        def done(bundle): self.install(bundle); self.dirty=True
        self.run(upload_and_open,done,'Загрузка КТ во временное хранилище...')
    def open_selected(self):
        items=self.history.selectedItems()
        if len(items)!=1: QMessageBox.information(self,'Выбор','Выберите одно исследование.'); return
        sid=items[0].data(Qt.UserRole); base=self.base()
        self.guard(lambda:self.run(lambda:unpack(http(base,'GET',f'/studies/{sid}/bundle').content),self.install))
    def predict(self):
        if not self.panel or self.job_id: return
        if self.panel.info['has_mask']:
            if QMessageBox.question(self,'Повторная обработка','Заменить рабочую маску новым предсказанием? Сохранённые версии останутся.',QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes: return
        sid=self.panel.info['id']; base=self.base()
        def done(v):
            self._session_has_uploads=True; self.job_id=v['job_id']; self.status.showMessage('Задача отправлена')
        self.run(lambda:http(base,'POST',f'/studies/{sid}/predict').json(),done)
    def poll(self):
        if not self.job_id or self.busy: return
        base=self.base(); jid=self.job_id; sid=self.panel.info['id']
        def fn():
            j=http(base,'GET',f'/jobs/{jid}').json()
            if j['status']=='done': j['bundle']=unpack(http(base,'GET',f'/studies/{sid}/bundle',params={'automatic':True}).content)
            return j
        def done(j):
            state={'queued':'В очереди','running':'Обработка','done':'Обработка завершена','error':'Ошибка','connection_error':'Нет связи с сервером'}.get(j['status'],j['status'])
            self.status.showMessage(f'{state}: {j["progress"]}%' + (' — '+j.get('error','') if j['status']=='connection_error' else ''))
            if j['status']=='error': self.job_id=None; QMessageBox.critical(self,'Сегментация',j.get('error','Ошибка'))
            elif j['status']=='done':
                self.job_id=None; _,mask,info=j['bundle']; self.panel.mask=mask
                self.panel.info.update(has_mask=True,model=info['model']); self.panel.refresh(); self.mark_mask_changed()
        def query():
            try: return fn()
            except Exception as e:
                if 'HTTP 404' in str(e): return {'status':'error','progress':0,'error':str(e)}
                return {'status':'connection_error','progress':0,'error':str(e)}
        self.run(query,done,label=None,block_ui=False)
    def mark_mask_changed(self):
        self.dirty=True; self.report_stale=bool(self.report.toPlainText().strip())
        if self.report_stale: self.status.showMessage('Маска изменена. Числа в заключении требуют проверки/обновления.')
    def edit(self):
        if not self.panel or self.job_id: return
        if not self.panel.info['has_mask']:
            if QMessageBox.question(self,'Пустая маска','Начать ручную разметку без модели?',QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes: return
        dialog=Editor(self,self.panel)
        dialog.setWindowState(dialog.windowState() | Qt.WindowMaximized)
        if dialog.exec_()==QDialog.Accepted:
            self.panel.mask=dialog.panel.mask; self.panel.info['has_mask']=True; self.panel.refresh(); self.mark_mask_changed()
    def generate(self):
        if not self.panel: return
        if self.job_id:
            QMessageBox.information(self,'Обработка','Дождитесь завершения обработки перед формированием заключения.'); return
        dialog=ReportDialog(self)
        dialog.exec_()

    def report_template(self):
        info=self.panel.info; meta=info['metadata']
        lines=[f'Исследование: {info["name"]}']
        for key,label in [('PatientID','Идентификатор пациента'),('StudyDate','Дата исследования'),('Modality','Модальность'),('Manufacturer','Производитель')]:
            if meta.get(key): lines.append(f'{label}: {meta[key]}')
        lines.extend(['',summary(info,self.panel.mask),'','Показатели рассчитаны по текущей маске и требуют проверки специалистом.','','Комментарий специалиста:',''])
        return '\n'.join(lines)
    def save(self,checked=False,after=None):
        if not self.panel or self.job_id or self.busy: return
        if self.report_stale:
            if QMessageBox.question(self,'Заключение','Маска менялась после создания текста. Сохранить без автоматического обновления заключения?',QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes: return
        info=self.panel.info; draft=self.is_draft(); folder_name=''
        if draft:
            original=info.get('original_name') or info['name']
            stem=original[:-7] if original.lower().endswith('.nii.gz') else Path(original).stem
            suggested=re.sub(r'[^A-Za-z0-9_-]','_',stem).strip('_-')[:64] or 'study_001'
            suggested=info.get('_proposed_name',suggested)
            while True:
                name,ok=QInputDialog.getText(self,'Первое сохранение',
                    'Название папки исследования (латиница, цифры, _ и -; до 64 символов):',text=suggested)
                if not ok: return
                name=name.strip(); suggested=name
                reserved={'CON','PRN','AUX','NUL'}|{f'{p}{i}' for p in ('COM','LPT') for i in range(1,10)}
                if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name) and name.upper() not in reserved:
                    folder_name=name; info['_proposed_name']=name; break
                QMessageBox.warning(self,'Название папки','Используйте латинские буквы, цифры, _ и -. Начните с буквы или цифры; зарезервированные имена Windows не допускаются.')
        sid=info['id']; base=self.base(); buffer=io.BytesIO()
        np.save(buffer,self.panel.mask,allow_pickle=False); payload=buffer.getvalue()
        data=dict(report=self.report.toPlainText(),model=info.get('model') or '',
                  has_mask=str(info['has_mask']).lower(),base_revision=str(info['revision']),folder_name=folder_name)
        def done(value):
            info.update(revision=value['revision'],metrics=value['metrics'],saved=True,
                        name=value.get('name',info['name']),storage_name=value.get('storage_name'))
            self.panel.title.setText(info['name']); self.dirty=False
            if after: after()
            else: self.refresh_history()
        self.run(lambda:http(base,'POST',f'/studies/{sid}/save',files={'mask':('mask.npy',payload)},data=data).json(),done,'Сохранение исследования...')
    def compare(self):
        items=self.history.selectedItems()
        if len(items)!=2: QMessageBox.information(self,'Выбор','Выберите два исследования с Ctrl.'); return
        names=[i.text() for i in items]
        first,ok=QInputDialog.getItem(self,'Порядок','Какое исследование считать «было»?',names,0,False)
        if not ok: return
        k=names.index(first); ids=[items[k].data(Qt.UserRole),items[1-k].data(Qt.UserRole)]; base=self.base()
        def done(bundles): Comparison(self,bundles).exec_()
        self.run(lambda:[unpack(http(base,'GET',f'/studies/{sid}/bundle').content) for sid in ids],done,'Загрузка сравнения...')
    def closeEvent(self,event):
        if self._close_authorized:
            self._finish_close(event); return
        if self.busy or self.job_id:
            QMessageBox.information(self,'Занято','Дождитесь завершения текущей задачи.')
            event.ignore(); return
        if not self.dirty:
            event.ignore(); self.cleanup_before_exit(); return
        message=QMessageBox(self); message.setWindowTitle('Изменения')
        message.setIcon(QMessageBox.Question)
        message.setText('Сохранить исследование на сервере перед выходом?' if self.is_draft() else 'Сохранить изменения на сервере перед выходом?')
        message.setInformativeText('Несохранённое новое исследование будет удалено из временного хранилища.' if self.is_draft() else 'Уже сохранённое исследование не будет удалено.')
        message.setStandardButtons(QMessageBox.Save|QMessageBox.Discard|QMessageBox.Cancel)
        message.button(QMessageBox.Save).setText('Сохранить')
        message.button(QMessageBox.Discard).setText('Не сохранять')
        message.button(QMessageBox.Cancel).setText('Отмена')
        message.setDefaultButton(QMessageBox.Save); message.setEscapeButton(QMessageBox.Cancel)
        answer=message.exec_(); event.ignore()
        if answer==QMessageBox.Cancel: return
        if answer==QMessageBox.Discard:
            self.cleanup_before_exit(); return
        self.save(after=self.cleanup_before_exit)

    def cleanup_before_exit(self):
        if self.busy: return
        if not self._session_has_uploads:
            self.dirty=False; self._close_authorized=True
            QTimer.singleShot(0,self.close); return
        base=self.base()
        def request():
            try:
                http(base,'DELETE','/session/temporary',timeout=(3,8)); return None
            except Exception as e: return str(e)
        def done(error):
            if error:
                answer=QMessageBox.question(self,'Очистка не подтверждена',
                    'Сервер не подтвердил очистку временных данных:\n'+error+
                    '\n\nЗакрыть приложение? При новой версии сервера остатки удалит автоматическая очистка после истечения срока хранения.',
                    QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
                if answer!=QMessageBox.Yes: return
            self.dirty=False; self._close_authorized=True
            QTimer.singleShot(0,self.close)
        self.run(request,done,'Очистка временных файлов перед выходом...')

    def _close_after_save(self):
        self.cleanup_before_exit()

    def _finish_close(self,event):
        self._closing=True; self.timer.stop(); self.health_timer.stop()
        if any(task.isRunning() for task in self.tasks):
            # Keep QThread owners alive until background requests finish.
            event.ignore(); self.hide(); QTimer.singleShot(100,self.close); return
        event.accept()

if __name__ == '__main__':
    if hasattr(Qt, 'AA_DisableWindowContextHelpButton'):
        QApplication.setAttribute(Qt.AA_DisableWindowContextHelpButton, True)
    app = QApplication(sys.argv)

    translator = QTranslator(app)

    translations_path = QLibraryInfo.location(
        QLibraryInfo.TranslationsPath
    )

    if translator.load('qtbase_ru', translations_path):
        app.installTranslator(translator)
    else:
        print(
            'Не найден перевод Qt: qtbase_ru.qm. '
            f'Папка переводов: {translations_path}'
        )

    window = Main()
    window.show()

    sys.exit(app.exec_())