"""
PVision AI: PDF Dataset Inventory & Origin Audit Generator
Creates a publication-quality PDF summarizing image counts, folder origins, resolutions, and ChatGPT prompt specifications.
"""

import sys
import os
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_dataset_inventory_pdf(output_pdf_path: str):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0f2942")     # Deep Navy
    c_secondary = colors.HexColor("#1b5e20")   # Forest Green
    c_accent = colors.HexColor("#d9534f")      # Coral/Red alert
    c_bg_light = colors.HexColor("#f8fafc")    # Off-white / light slate
    c_border = colors.HexColor("#cbd5e1")      # Slate border
    c_text = colors.HexColor("#1e293b")        # Dark slate text

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=c_primary,
        alignment=0,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=c_text
    )

    body_bold = ParagraphStyle(
        "Body_Bold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a")
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=c_text
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold"
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=table_cell_style,
        alignment=1
    )

    story = []

    # Title & Metadata Header
    story.append(Paragraph("PVision AI: Photovoltaic Fault Image Dataset Audit", title_style))
    story.append(Paragraph("Complete Inventory of Image Sample Counts, Folder Origins, and Modality Breakdown for AI / ChatGPT Handover", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=0, spaceAfter=10))

    # Executive Overview Key Metrics Cards
    kpi_data = [
        [
            Paragraph("<b>TOTAL IMAGES</b><br/><font size='14'><b>69,484</b></font><br/>Across 2 Modalities", table_cell_center),
            Paragraph("<b>GASF MODALITY</b><br/><font size='14'><b>35,000</b></font><br/>256x256 RGB", table_cell_center),
            Paragraph("<b>I-V MODALITY</b><br/><font size='14'><b>34,484</b></font><br/>1343x808 RGB", table_cell_center),
            Paragraph("<b>FAULT CLASSES</b><br/><font size='14'><b>7 Classes</b></font><br/>1 Normal + 6 Faults", table_cell_center),
            Paragraph("<b>DATA INTEGRITY</b><br/><font size='14'><b>100%</b></font><br/>0 Corrupt, 0 Dups", table_cell_center)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[108, 108, 108, 108, 108])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 12))

    # Section 1: Overview
    story.append(Paragraph("1. Primary Storage Locations & Folder Hierarchy", h1_style))
    story.append(Paragraph(
        "The project dataset resides physically on disk under the following root paths. Each modality is partitioned into seven distinct subfolders named strictly after the physical fault category.",
        body_style
    ))
    story.append(Spacer(1, 4))

    paths_data = [
        [Paragraph("<b>Directory Role</b>", table_header_style), Paragraph("<b>Physical Origin Path on Disk</b>", table_header_style), Paragraph("<b>Image Modality</b>", table_header_style)],
        [Paragraph("Raw Dataset Root", table_cell_bold), Paragraph("<code>D:\\SolarPredict\\dataset\\gasf_images</code>", code_style), Paragraph("GASF (Gramian Angular Field)", table_cell_style)],
        [Paragraph("Raw Dataset Root", table_cell_bold), Paragraph("<code>D:\\SolarPredict\\dataset\\iv_images</code>", code_style), Paragraph("I-V Characteristic Curves", table_cell_style)],
        [Paragraph("Project Data Root", table_cell_bold), Paragraph("<code>D:\\SolarPredict\\PVision_AI\\data\\pv_fault\\gasf_images</code>", code_style), Paragraph("GASF (Mirrored / Linked)", table_cell_style)],
        [Paragraph("Project Data Root", table_cell_bold), Paragraph("<code>D:\\SolarPredict\\PVision_AI\\data\\pv_fault\\iv_images</code>", code_style), Paragraph("I-V Curves (Mirrored / Linked)", table_cell_style)],
    ]
    t_paths = Table(paths_data, colWidths=[110, 310, 120])
    t_paths.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_paths)
    story.append(Spacer(1, 12))

    # Section 2: Complete Folder Breakdown Table
    story.append(Paragraph("2. Detailed Folder Origin & Image Count Breakdown", h1_style))
    story.append(Paragraph(
        "Below is the exact audit of every individual folder origin, sample count, percentage share, native dimensions, and sequence validation status:",
        body_style
    ))
    story.append(Spacer(1, 4))

    table_data = [
        [
            Paragraph("<b>Modality</b>", table_header_style),
            Paragraph("<b>Class Name</b>", table_header_style),
            Paragraph("<b>Folder Origin (Relative to dataset/)</b>", table_header_style),
            Paragraph("<b>Image Count</b>", table_header_style),
            Paragraph("<b>Share</b>", table_header_style),
            Paragraph("<b>Resolution</b>", table_header_style),
            Paragraph("<b>Audit Status</b>", table_header_style)
        ]
    ]

    gasf_classes = [
        ("crack", 5000, "gasf_images/crack", "14.29%", "256x256 RGB", "Complete (1-5000)"),
        ("global_aging", 5000, "gasf_images/global_aging", "14.29%", "256x256 RGB", "Complete (1-5000)"),
        ("hotspot", 5000, "gasf_images/hotspot", "14.29%", "256x256 RGB", "Complete (1-5000)"),
        ("normal", 5000, "gasf_images/normal", "14.29%", "256x256 RGB", "Complete (1-5000)"),
        ("partial_aging", 5000, "gasf_images/partial_aging", "14.29%", "256x256 RGB", "Complete (1-5000)"),
        ("shading", 5000, "gasf_images/shading", "14.29%", "256x256 RGB", "Complete (1-5000)"),
        ("short_circuit", 5000, "gasf_images/short_circuit", "14.29%", "256x256 RGB", "Complete (1-5000)"),
    ]

    for cls, count, folder, share, res, status in gasf_classes:
        table_data.append([
            Paragraph("GASF", table_cell_bold),
            Paragraph(f"<b>{cls}</b>", table_cell_style),
            Paragraph(f"<code>{folder}</code>", code_style),
            Paragraph(f"{count:,}", table_cell_center),
            Paragraph(share, table_cell_center),
            Paragraph(res, table_cell_center),
            Paragraph(f"<font color='#166534'>{status}</font>", table_cell_style),
        ])

    table_data.append([
        Paragraph("<b>GASF Subtotal</b>", table_cell_bold),
        Paragraph("<b>7 Classes</b>", table_cell_bold),
        Paragraph("<b>gasf_images/ (7 folders)</b>", table_cell_bold),
        Paragraph("<b>35,000</b>", table_cell_center),
        Paragraph("<b>50.37%</b>", table_cell_center),
        Paragraph("<b>256x256</b>", table_cell_center),
        Paragraph("<b>100% Balanced</b>", table_cell_style),
    ])

    iv_classes = [
        ("crack", 5000, "iv_images/crack", "14.50%", "1343x808 RGB", "Complete (1-5000)"),
        ("global_aging", 5000, "iv_images/global_aging", "14.50%", "1343x808 RGB", "Complete (1-5000)"),
        ("hotspot", 5000, "iv_images/hotspot", "14.50%", "1343x808 RGB", "Complete (1-5000)"),
        ("normal", 5000, "iv_images/normal", "14.50%", "1343x808 RGB", "Complete (1-5000)"),
        ("partial_aging", 5000, "iv_images/partial_aging", "14.50%", "1343x808 RGB", "Complete (1-5000)"),
        ("shading", 5000, "iv_images/shading", "14.50%", "1343x808 RGB", "Complete (1-5000)"),
        ("short_circuit", 4484, "iv_images/short_circuit", "13.00%", "1343x808 RGB", "516 files missing (1-513, 597-599)"),
    ]

    for cls, count, folder, share, res, status in iv_classes:
        stat_color = "#991b1b" if "missing" in status else "#166534"
        table_data.append([
            Paragraph("I-V Curves", table_cell_bold),
            Paragraph(f"<b>{cls}</b>", table_cell_style),
            Paragraph(f"<code>{folder}</code>", code_style),
            Paragraph(f"{count:,}", table_cell_center),
            Paragraph(share, table_cell_center),
            Paragraph(res, table_cell_center),
            Paragraph(f"<font color='{stat_color}'>{status}</font>", table_cell_style),
        ])

    table_data.append([
        Paragraph("<b>I-V Subtotal</b>", table_cell_bold),
        Paragraph("<b>7 Classes</b>", table_cell_bold),
        Paragraph("<b>iv_images/ (7 folders)</b>", table_cell_bold),
        Paragraph("<b>34,484</b>", table_cell_center),
        Paragraph("<b>49.63%</b>", table_cell_center),
        Paragraph("<b>1343x808</b>", table_cell_center),
        Paragraph("<b>4,484 short_circuit</b>", table_cell_style),
    ])

    table_data.append([
        Paragraph("<b>GRAND TOTAL</b>", table_cell_bold),
        Paragraph("<b>7 Unique Classes</b>", table_cell_bold),
        Paragraph("<b>14 Total Folder Origins</b>", table_cell_bold),
        Paragraph("<b>69,484</b>", table_cell_center),
        Paragraph("<b>100.0%</b>", table_cell_center),
        Paragraph("<b>RGB PNG</b>", table_cell_center),
        Paragraph("<b>0 Corrupt, 0 Dups</b>", table_cell_style),
    ])

    t_main = Table(table_data, colWidths=[65, 75, 140, 55, 45, 75, 85])
    t_main.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-3), [colors.white, c_bg_light]),
        ('BACKGROUND', (0,8), (-1,8), colors.HexColor("#e2e8f0")),  # GASF Subtotal
        ('BACKGROUND', (0,16), (-1,16), colors.HexColor("#e2e8f0")), # IV Subtotal
        ('BACKGROUND', (0,17), (-1,17), colors.HexColor("#cbd5e1")), # Grand Total
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_main)
    story.append(Spacer(1, 14))

    # Section 3: Summary text ready for ChatGPT handover
    story.append(Paragraph("3. ChatGPT Ready-to-Copy Specification Block", h1_style))
    story.append(Paragraph(
        "Copy and paste the formatted block below directly into ChatGPT to provide complete context of the image dataset:",
        body_style
    ))
    story.append(Spacer(1, 4))

    chatgpt_prompt_text = (
        "Dataset Specification for PVision AI Fault Classification:<br/>"
        "• Total Images: 69,484 PNG files across 14 directory origins.<br/>"
        "• Modality 1 (GASF Images): 35,000 images, 256x256 RGB, 7 balanced classes (5,000 each: crack, global_aging, hotspot, normal, partial_aging, shading, short_circuit). Origin: D:\\SolarPredict\\dataset\\gasf_images\\<class_name><br/>"
        "• Modality 2 (I-V Curves): 34,484 images, 1343x808 RGB, 7 classes (5,000 for 6 classes, 4,484 for short_circuit due to 516 missing index sequence numbers: 1-513, 597-599). Origin: D:\\SolarPredict\\dataset\\iv_images\\<class_name><br/>"
        "• Data Quality: 0 corrupted files, 0 duplicate images (MD5 verified), 100% sample retention.<br/>"
        "• Target Preprocessing: Resize to 224x224 RGB, ImageNet standardization, 70% Train (48,638), 15% Val (10,423), 15% Test (10,423) stratified split.<br/>"
        "• Objective: Train a Convolutional Neural Network (e.g. ResNet-18 or MobileNetV3) for 7-class PV fault diagnosis."
    )

    t_prompt = Table([[Paragraph(chatgpt_prompt_text, code_style)]], colWidths=[540])
    t_prompt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#94a3b8")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_prompt)

    doc.build(story)
    print(f"Successfully generated PDF at: {output_pdf_path}")

if __name__ == "__main__":
    out_pdf = "d:/SolarPredict/PVision_AI_Image_Dataset_Audit.pdf"
    create_dataset_inventory_pdf(out_pdf)
