
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os
from datetime import date

activeDirectoryName = "dataset"
currentDirectory = os.getcwd()
path = os.path.join(currentDirectory, activeDirectoryName)

# What things should be inside the report?
# There should be two sections: Dataset and model

#Create a word prototype

# Dataset
"""
* +|Dataset section should hold inforation about how many repetitions/segments are there. Get from shape
* +|How many samples per segment are there. get from shape
* x|Also could include information about each of the set file. new function
* +|What is the sensor freaquency, what does it mean for sensors. new function
* x|Infromation about each of the forms and their repetitions. new function
"""
# Model
"""
* x|Mentioned metrics and explanation
* x|Training time
"""

def create_report(total_records, sensor_freq, segment_length, form_counts, file_squat_data):
    document = Document()
    
    # Title
    title = document.add_heading('ATSKAITE', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Subtitle
    subtitle = document.add_paragraph('PIETUPIENA DATUKOPAI UN KLASIFIKĀCIJAI')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Date
    document.add_paragraph(str(date.today()), style='Normal')

    # Description
    document.add_paragraph(
        'Dokumenta apraksts\n'
        '\tDokuments sastāv no divām daļām: datu kopa un modelis. Datu kopas daļā tiek aprakstītas dažādas specifikas par izveidoto datu kopu. '
        'Pretēji tas pats arī ir darīts modeļa nodaļā.\n'
    )

    # Section: DATU KOPA
    document.add_heading('DATU KOPA', level=1)

    # Dataset Overview
    document.add_paragraph('Datu kopas, kopējās iezīmes')
    document.add_paragraph(f'Kopējais segmentu/pietupienu/ierakstu skaits – {total_records}')
    document.add_paragraph(f'Sensora frekvence - {sensor_freq}')
    document.add_paragraph(f'Segmentu garums – {segment_length}')

    # Table: Form counts. Key value pairs
    document.add_paragraph('Formu iedalījums ierakstu daudzumos:')
    table1 = document.add_table(rows=1, cols=2)
    table1.style = 'Table Grid'
    hdr_cells = table1.rows[0].cells
    hdr_cells[0].text = 'Formas nosaukums'
    hdr_cells[1].text = 'Ierakstu daudzums'
    for form, count in form_counts.items():
        row_cells = table1.add_row().cells
        row_cells[0].text = str(form)
        row_cells[1].text = str(count)

    # Dataset specifics
    document.add_paragraph('\nDatu kopas, piegājienu iezīmes')
    document.add_paragraph('Katra faila pietupienu daudzums')

    # Table: File squat data. Key value pairs
    table2 = document.add_table(rows=1, cols=3)
    table2.style = 'Table Grid'
    hdr_cells = table2.rows[0].cells
    hdr_cells[0].text = 'Faila nosaukums'
    hdr_cells[1].text = 'Pietupieni pirms pirmsprocesēšanas'
    hdr_cells[2].text = 'Pietupieni pēc pirmsprocesēšanas'
    for file_data in file_squat_data:
        row_cells = table2.add_row().cells
        row_cells[0].text = file_data['filename']
        row_cells[1].text = str(file_data['before'])
        row_cells[2].text = str(file_data['after'])

    # Save the document
    document.save(os.path.join(path, "report.docx"))

# create_report()