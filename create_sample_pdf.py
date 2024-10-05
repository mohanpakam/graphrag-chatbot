from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import random

def create_sample_pdf(filename):
    doc = SimpleDocTemplate(filename, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    elements = []

    # Custom styles
    if 'Heading2' not in styles:
        styles.add(ParagraphStyle(name='Heading2', fontSize=14, spaceBefore=12, spaceAfter=6))
    if 'BodyText' not in styles:
        styles.add(ParagraphStyle(name='BodyText', fontSize=12, spaceBefore=6, spaceAfter=6))

    # Title
    elements.append(Paragraph("Complex PDF with Side-by-Side Tables and Multi-Page Table", styles['Title']))
    elements.append(Spacer(1, 12))

    # Introduction
    elements.append(Paragraph("This document contains multiple tables with varying structures placed side by side with different spacing between them, as well as a large table spanning multiple pages without repeating headers.", styles['BodyText']))
    elements.append(Spacer(1, 12))

    # Create three tables with different structures
    table1_data = [['Table 1', 'Col 1', 'Col 2']] + [[f'Row {i}', random.randint(1, 100), random.randint(1, 100)] for i in range(1, 6)]
    table2_data = [['Table 2', 'A', 'B', 'C']] + [[f'Row {i}', random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)] for i in range(1, 8)]
    table3_data = [['Table 3', 'X', 'Y', 'Z', 'W']] + [[f'Row {i}', random.randint(1, 100), random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)] for i in range(1, 10)]

    table1 = Table(table1_data, colWidths=[1*inch, 0.8*inch, 0.8*inch])
    table2 = Table(table2_data, colWidths=[1*inch, 0.7*inch, 0.7*inch, 0.7*inch])
    table3 = Table(table3_data, colWidths=[1*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch])

    for table in [table1, table2, table3]:
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

    # Create spacers with varying widths
    spacer1 = Spacer(0.5*inch, 1)  # 0.5 inch spacer
    spacer2 = Spacer(1.2*inch, 1)  # 1.2 inch spacer

    # Create a table to hold the three tables with varying spaces between them
    main_table = Table([
        [table1, spacer1, table2, spacer2, table3]
    ], colWidths=[2.6*inch, 0.5*inch, 3.1*inch, 1.2*inch, 3.4*inch])

    main_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    elements.append(main_table)

    # Add some text after the tables
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("This text appears after the side-by-side tables to provide additional context and complexity to the document layout.", styles['BodyText']))
    elements.append(Spacer(1, 12))

    # Create a large table that spans multiple pages
    large_table_data = [['ID', 'Name', 'Value 1', 'Value 2', 'Value 3', 'Value 4', 'Value 5']]
    for i in range(1, 201):  # 200 rows to ensure it spans multiple pages
        large_table_data.append([
            i,
            f'Item {i}',
            random.randint(1000, 9999),
            random.randint(1000, 9999),
            random.randint(1000, 9999),
            random.randint(1000, 9999),
            random.randint(1000, 9999)
        ])

    # Create the large table without repeating headers
    large_table = Table(large_table_data, colWidths=[0.5*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    large_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    elements.append(Paragraph("Large Multi-Page Table", styles['Heading2']))
    elements.append(large_table)

    # Build the PDF
    doc.build(elements)

if __name__ == "__main__":
    create_sample_pdf("complex_sample.pdf")
    print("Complex sample PDF with side-by-side tables and multi-page table created successfully.")