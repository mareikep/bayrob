import pyrap
from pyrap.constants import DLG
from pyrap.layout import ColumnLayout, RowLayout
from pyrap.themes import DisplayTheme
from pyrap.widgets import Shell, Composite, Label, Edit, Button, constructor


class MinMaxBox(Shell):
    '''
    Represents a simple message box containing two edit fields for text input.
    '''

    @constructor('MinMaxBox')
    def __init__(self, parent, title, unit=None, min=None, max=None, icon=None, message=False, multiline=False, password=True, modal=True, resize=False,
                 btnclose=True):
        Shell.__init__(self, parent=parent, title=title, titlebar=True, border=True,
                       btnclose=btnclose, resize=resize, modal=modal)
        self.icontheme = DisplayTheme(self, pyrap.session.runtime.mngr.theme)
        self.icon = {DLG.INFORMATION: self.icontheme.icon_info,
                     DLG.QUESTION: self.icontheme.icon_question,
                     DLG.WARNING: self.icontheme.icon_warning,
                     DLG.ERROR: self.icontheme.icon_error}.get(icon)
        self.answer = None
        self.message = message
        self.multiline = multiline
        self.password = password
        self.unit = unit
        self.min = min
        self.max = max

    def answer_and_close(self, a) -> None:
        self.answer = a
        self.close()

    def create_content(self) -> None:
        Shell.create_content(self)
        mainarea = Composite(self.content)
        mainarea.layout = ColumnLayout(padding=10, hspace=5)
        img = None
        if self.icon is not None:
            img = Label(mainarea, img=self.icon, valign='top', halign='left')

        textarea = Composite(mainarea)
        textarea.layout = RowLayout()
        Label(textarea, text=self.message, valign='fill', halign='fill')
        self.inputfieldunit = Edit(textarea, text=self.unit, message='Enter unit value', multiline=self.multiline, password=self.password, valign='fill', halign='fill')
        self.inputfieldmin = Edit(textarea, text=self.min, message='Enter min value', multiline=self.multiline, password=self.password, valign='fill', halign='fill')
        self.inputfieldmax = Edit(textarea, text=self.max, message='Enter max value', multiline=self.multiline, password=self.password, valign='fill', halign='fill')

        buttons = Composite(textarea)
        buttons.layout = ColumnLayout(equalwidths=True, halign='right',
                                      valign='bottom')
        self.create_buttons(buttons)

    def create_buttons(self, buttons) -> None:
        ok = Button(buttons, text='OK', minwidth=100)
        ok.on_select += lambda *_: self.answer_and_close([self.inputfieldunit.text, self.inputfieldmin.text, self.inputfieldmax.text])
        cancel = Button(buttons, text='Cancel', halign='fill')
        cancel.on_select += lambda *_: self.answer_and_close(None)
