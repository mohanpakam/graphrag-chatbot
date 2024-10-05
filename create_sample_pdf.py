from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import random

def create_sample_pdf(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Custom styles
    if 'Heading2' not in styles:
        styles.add(ParagraphStyle(name='Heading2', fontSize=14, spaceBefore=12, spaceAfter=6))
    else:
        styles.add(ParagraphStyle(name='CustomHeading2', fontSize=14, spaceBefore=12, spaceAfter=6))
    
    if 'BodyText' not in styles:
        styles.add(ParagraphStyle(name='BodyText', fontSize=12, spaceBefore=6, spaceAfter=6))
    else:
        styles.add(ParagraphStyle(name='CustomBodyText', fontSize=12, spaceBefore=6, spaceAfter=6))

    heading_style = styles['Heading2'] if 'Heading2' in styles else styles['CustomHeading2']
    body_style = styles['BodyText'] if 'BodyText' in styles else styles['CustomBodyText']

    # Title
    elements.append(Paragraph("Complex PDF with Tables and Explanations", styles['Title']))
    elements.append(Spacer(1, 12))

    # Introduction
    elements.append(Paragraph("This document contains multiple tables with explanatory text and summaries. Some tables span across multiple pages.", body_style))
    elements.append(Spacer(1, 12))

    # Table 1: Simple table with explanation
    elements.append(Paragraph("1. Employee Information", heading_style))
    elements.append(Paragraph("The following table shows basic employee information including their name, age, and city of residence.", body_style))
    
    data = [
        ['Name', 'Age', 'City'],
        ['Alice Johnson', '28', 'New York'],
        ['Bob Smith', '35', 'London'],
        ['Charlie Brown', '42', 'Paris'],
        ['Diana Miller', '31', 'Tokyo']
    ]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Summary: This table provides a quick overview of our international team, showcasing the diversity in age and location of our employees.", body_style))
    elements.append(Spacer(1, 12))

    # Table 2: Large table spanning multiple pages
    elements.append(Paragraph("2. Detailed Sales Report", heading_style))
    elements.append(Paragraph("The following table presents a comprehensive sales report for the past year, broken down by month and product category. Due to its size, this table may span multiple pages.", body_style))
    
    # Generate sample data for a large table
    headers = ['Month', 'Electronics', 'Clothing', 'Food', 'Books', 'Total']
    large_data = [headers]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month in months:
        row = [month] + [f"${random.randint(1000,9999)}" for _ in range(4)]
        row.append(f"${sum(int(x[1:]) for x in row[1:])}")
        large_data.append(row)
    
    large_table = Table(large_data, repeatRows=1)
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
    elements.append(large_table)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Summary: This comprehensive sales report illustrates our company's performance across different product categories throughout the year. It helps identify trends and seasonal variations in our sales data.", body_style))
    elements.append(PageBreak())

    # Table 3: Nested table with explanation
    elements.append(Paragraph("3. Department Structure", heading_style))
    elements.append(Paragraph("The following table shows our company's department structure, including the head of each department and the number of employees. This table demonstrates a more complex structure with nested information.", body_style))
    
    nested_data = [
        ['Department', 'Details'],
        ['HR', Table([['Head', 'John Doe'], ['Employees', '15']])],
        ['Finance', Table([['Head', 'Jane Smith'], ['Employees', '20']])],
        ['IT', Table([['Head', 'Mike Johnson'], ['Employees', '30']])],
        ['Marketing', Table([['Head', 'Sarah Brown'], ['Employees', '25']])]
    ]
    nested_table = Table(nested_data)
    nested_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(nested_table)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Summary: This nested table provides a clear overview of our company's departmental structure, highlighting the leadership and size of each key department. It showcases the distribution of our workforce across different functional areas.", body_style))

    # Build the PDF
    doc.build(elements)

if __name__ == "__main__":
    create_sample_pdf("complex_sample.pdf")
    print("Complex sample PDF created successfully.")