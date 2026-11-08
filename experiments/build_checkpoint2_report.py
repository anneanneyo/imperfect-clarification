"""Build the checkpoint report from checked local artifacts; bundled reportlab.

Run after checkpoint2_eda.py. No network calls and no model training.
"""
from pathlib import Path
import json
import os
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/checkpoint2_2026-10-27'
PDF=ROOT/'output/pdf/checkpoint2_eda_2026-10-27.pdf'
font_dirs=[Path(os.environ.get('CHECKPOINT_FONT_DIR','/usr/share/fonts/truetype/dejavu')),
           Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/fonts',
           Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype']
try:
    import matplotlib
    font_dirs.append(Path(matplotlib.get_data_path())/'fonts/ttf')
except ImportError:
    pass
def font_file(name):
    for directory in font_dirs:
        if (directory/name).exists():return directory/name
    raise FileNotFoundError('Install DejaVu Sans fonts or set CHECKPOINT_FONT_DIR to their directory')
pdfmetrics.registerFont(TTFont('DV',str(font_file('DejaVuSans.ttf'))))
bold=font_file('DejaVuSans-Bold.ttf')
pdfmetrics.registerFont(TTFont('DV-Bold',str(bold)))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')
NAVY=colors.HexColor('#18354B'); TEAL=colors.HexColor('#147A79'); GRAY=colors.HexColor('#57636D')
styles={
 'body':ParagraphStyle('body',fontName='DV',fontSize=10.5,leading=15.3,spaceAfter=8,textColor=NAVY),
 'small':ParagraphStyle('small',fontName='DV',fontSize=8.4,leading=11.9,spaceAfter=6,textColor=GRAY),
 'h1':ParagraphStyle('h1',fontName='DV-Bold',fontSize=20,leading=26,spaceAfter=15,textColor=NAVY),
 'h2':ParagraphStyle('h2',fontName='DV-Bold',fontSize=13,leading=18,spaceBefore=9,spaceAfter=9,textColor=TEAL),
 'title':ParagraphStyle('title',fontName='DV-Bold',fontSize=27,leading=34,spaceAfter=16,textColor=NAVY),
}


def P(s,style='body'):return Paragraph(s,styles[style])


def table(rows,widths):
    t=Table([[P(escape(str(c)),'small') for c in row] for row in rows],colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4F1F1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.6,TEAL),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#DCE3E7'))]))
    return t


def figure(path,caption,maxheight=280):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:w,h=im.size
    width=483; height=width*h/w
    if height>maxheight:width*=maxheight/height;height=maxheight
    return [Image(str(path),width=width,height=height,hAlign='CENTER'),P(caption,'small')]


def page_header(c,doc):
    c.setFont('DV',8);c.setFillColor(GRAY)
    c.drawString(56,811,'Годовой проект / Разведочный анализ данных / 27 октября 2026')
    c.setStrokeColor(colors.HexColor('#DCE3E7'));c.line(56,802,539,802)
    c.drawString(56,28,'Губарева Анна Владимировна | Чекпойнт 2')
    c.drawRightString(539,28,str(doc.page))


def build():
    # Section content and figure list are kept in a readable, editable JSON spec.
    spec=json.loads((OUT/'report_spec.json').read_text())
    story=[]
    for n,page in enumerate(spec['pages']):
        if n:story.append(PageBreak())
        for item in page:
            kind=item['type']
            if kind=='text':story.append(P(item['text'],item.get('style','body')))
            elif kind=='space':story.append(Spacer(1,item['height']))
            elif kind=='table':story.append(table(item['rows'],item['widths']))
            elif kind=='figure':story.extend(figure(OUT/item['path'],item['caption'],item.get('maxheight',280)))
    PDF.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=56,rightMargin=56,topMargin=58,bottomMargin=50,title='Когда уточнение помогает: разведочный анализ данных',author='Губарева Анна Владимировна')
    doc.build(story,onFirstPage=page_header,onLaterPages=page_header)
    print(PDF)


if __name__=='__main__':build()
