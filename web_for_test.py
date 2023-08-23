import pytesseract
import cv2
from datetime import datetime
import os
import re
import zipfile
import pandas as pd
import streamlit as st
import numpy as np
import io


st.set_page_config(
    page_title="Cert",
    layout = 'wide',
)

st.title('Certicatate Scraper')

# Initialize the Tesseract OCR
def initialize_tesseract():
    pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# Initialize Tesseract
initialize_tesseract()

def process_cropped_images(img, coordinates):
    result = []

    for idx, (upper, lower, left, right) in enumerate(coordinates):
        crop = img[upper:lower, left:right]

        # Perform OCR on the cropped image
        extracted_text = pytesseract.image_to_string(crop)

        result.append(extracted_text.strip())

    return result

def process_cropped_images1(img, coordinates):
    # Process cropped images and get the extracted text
    extracted_texts = process_cropped_images(img, coordinates)
    
    split_results = []
    for text in extracted_texts:
        split_result = text.split('\n')
        split_results.append(split_result)

    data = [[item for item in split_result if item != ''] for split_result in split_results]
    # Extract values and keys
    values = [item.split(' ', 1)[1].strip() for item in data[0]]
    keys = [item.split(' ', 1)[0].strip() for item in data[0]]

    # Create a dictionary from keys and values
    data_dict = dict(zip(keys, values))

    # Create a DataFrame
    df = pd.DataFrame([data_dict])
        
    return df


def extract_origin_info(image_path, coordinates):

    # Process cropped images and get the extracted text
    extracted_texts = process_cropped_images(img, coordinates)

    # Extract "Origin" information
    origin_info = None
    for text in extracted_texts:
            if 'Origin' in text:
                origin_info = text.split(':', 1)[1].strip()

    # Create a DataFrame with "Origin" as a column header and the extracted data as data
    data = {'Origin': [origin_info]}
    df = pd.DataFrame(data)

    return df

def extrace_img3(img, coordinates):
    # Process the cropped image and extract text
    extracted_texts = process_cropped_images(img, coordinates)

    # Initialize the list to store extracted values
    list1 = []

    # Process each extracted text
    for text in extracted_texts:
        lines = text.split('\n')
        for line in lines:
            if line.strip() and any(char.isalpha() for char in line):
                list1.append(line.strip())
    
    header = list1[:6]
    header.append('comment')
    data_header = list1[6:]
    
    df = pd.DataFrame([data_header], columns=header)
    df['comment'] = df['Comment'] + ' ' + df['comment']
    df['Dimensions'] = df['Dimensions'].str.replace('$', '5', regex=False)
    df = df.drop(columns=['comment'])
    
    return df

def detect_color(text):
    if "PigeonsBlood" in text:
        return "PigeonsBlood"
    elif "*" in text:
        return "PigeonsBlood"
    elif "pigeon-blood" in text:
        return "PigeonsBlood"
    else:
        return None
    
def detect_cut(cut):
    if cut != "Cabochon":
        return "cut"
    else:
        return cut
    
def detect_shape(shape):
    valid_shapes = [
        "cushion", "heart", "marquise", "octagonal", "oval",
        "pear", "rectangular", "round", "square", "triangular",
        "star", "sugarloaf", "tumbled"
    ]
    if shape in valid_shapes:
        return shape
    else:
        return "Others"
    
def detect_origin(origin):
    if not origin.strip():
        return "No origin"
    
    # Remove words in parentheses
    origin_without_parentheses = re.sub(r'\([^)]*\)', '', origin)
    return origin_without_parentheses.strip()

def reformat_issued_date(issued_date):
    try:
        # Remove ordinal suffixes (e.g., "th", "nd", "rd")
        cleaned_date = re.sub(r'(?<=\d)(st|nd|rd|th)\b', '', issued_date)

        # Parse the cleaned date string
        parsed_date = datetime.strptime(cleaned_date, '%d %B %Y')

        # Reformat the date to YYYY-MM-DD
        reformatted_date = parsed_date.strftime('%Y-%m-%d')
        return reformatted_date
    except ValueError:
        return ""
    
def detect_mogok(origin):
    return str("(Mogok, Myanmar)" in origin)

def generate_indication(comment):
    if comment in ["H", "H(a)", "H(b)", "H(c)"]:
        return "Heated"
    elif comment == "No indication of thermal treatment":
        return "Unheated"
    else:
        return "Unknown"
    
def detect_old_heat(comment, indication):
    if indication == "Heated":
        if comment in ["H", "H(a)", "H(b)", "H(c)"]:
            return "oldHeat"
    return "Others"

def generate_display_name(color, origin):
    display_name = ""

    if "*" in color:
        display_name = "PGB*"
    elif color == "Red":
        display_name = "GRS"
    elif color == "PigeonsBlood":
        display_name = "PGB"
    
    if "(Mogok, Myanmar)" in origin:
        display_name = "MG-" + display_name
    
    return display_name

certificate_acronyms = {
    "GRS": "GRS",
    "SSEF": "SSEF",
    "GBL": "GBL",
    "CDC": "CDC",
    "AIGS": "AIGS",
    "AGL" : "AGL",
    "BELL" : "BELL",
    "GCI" : "GCI",
    "GIT" : "GIT",
    "GGT" : "GGT",
    "ICL" : "ICL",
    "LOTUS" : "LOTUS",
    "ICA" : "ICA",
}

# Define the function to extract the year and number from certNO
def extract_cert_info(certNO):
    parts = certNO.split("-")
    if len(parts) == 2:
        cert_name = parts[0][:-4]  # Extract the first part of the certificate name (e.g., "GRS")
        cert_number = parts[0][-4:] + "-" + parts[1]  # Extract the last four characters and combine with the year
        return cert_name, cert_number
    return "", ""

def convert_carat_to_numeric(value_with_unit):
    value_without_unit = value_with_unit.replace(" ct", "")
    numeric_value = float(value_without_unit)
    return numeric_value

def convert_dimension(dimension_str):
    parts = dimension_str.split(" x ")
    if len(parts) == 3 and parts[-1].endswith(" (mm)"):
        length = float(parts[0])
        width = float(parts[1])
        height = float(parts[2][:-5])  # Remove " (mm)" from the last part
        return length, width, height
    return None, None, None

def rename_identification_to_stone(dataframe):
    dataframe.rename(columns={"Identification": "Stone"}, inplace = True)
    return dataframe

# Define the function to perform all data processing steps
def perform_data_processing(result_df):
    
    result_df["Detected_Color"] = result_df["Color"].apply(detect_color)
    result_df["Detected_Cut"] = result_df["Cut"].apply(detect_cut)
    result_df["Detected_Shape"] = result_df["Shape"].apply(detect_shape)
    result_df["Detected_Origin"] = result_df["Origin"].apply(detect_origin)
    result_df["Reformatted_issuedDate"] = result_df["Date"].apply(reformat_issued_date)
    result_df["Mogok"] = result_df["Origin"].apply(detect_mogok)
    result_df["Indication"] = result_df["Comment"].apply(generate_indication)
    result_df["oldHeat"] = result_df.apply(lambda row: detect_old_heat(row["Comment"], row["Indication"]), axis=1)
    result_df["displayName"] = result_df.apply(lambda row: generate_display_name(row["Color"], row["Origin"]), axis=1)
    result_df["certName"], result_df["certNO"] = zip(*result_df["No."].apply(extract_cert_info))
    result_df["carat"] = result_df["Weight"].apply(convert_carat_to_numeric)
    result_df[["length", "width", "height"]] = result_df["Dimensions"].apply(convert_dimension).apply(pd.Series)
    result_df = rename_identification_to_stone(result_df)

    result_df = result_df[[
    "certName",
    "certNO",
    "displayName",
    "Stone",
    "Detected_Color",
    "Detected_Origin",
    "Reformatted_issuedDate",
    "Indication",
    "oldHeat",
    "Mogok",
    "Detected_Cut",
    "Detected_Shape",
    "carat",
    "length",
    "width",
    "height"
    ]]
    
    return result_df




# Define cropping coordinates for different regions
coordinates = [
    (570, 910, 168, 1010),
    (550, 980, 1800, 3000),
    (1580, 2180, 175, 1400) 
]

# Specify the folder containing the images
# folder_path = r'C:\Users\kan43\Downloads\Cert Scraping Test'

# Specify the file pattern you want to filter
file_pattern = "-01_GRS"

# Create a Streamlit file uploader for the zip file
zip_file = st.file_uploader("Upload a ZIP file containing images", type=["zip"])

if zip_file is not None:
    # Extract the uploaded ZIP file
    with zipfile.ZipFile(zip_file) as zip_data:
        df_list = []

        for image_file in zip_data.namelist():
            if file_pattern in image_file:
                filename_without_suffix = image_file.split('-')[0]
                # Read the image
                with zip_data.open(image_file) as file:
                    img_data = io.BytesIO(file.read())
                    img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                    
                    # Process the image and perform data processing
                    df_1 = process_cropped_images1(img, [coordinates[0]])
                    df_2 = extract_origin_info(img, [coordinates[1]])
                    df_3 = extrace_img3(img, [coordinates[2]])
                    result_df = pd.concat([df_1, df_2, df_3], axis=1)
                    result_df = perform_data_processing(result_df)
                    result_df['StoneID'] = filename_without_suffix
                    result_df["StoneID"] = result_df["StoneID"].str.split("/")
                    # Get the last part of each split
                    result_df["StoneID"] = result_df["StoneID"].str.get(-1)

                    result_df = result_df[[
                        "certName",
                        "certNO",
                        "StoneID",
                        "displayName",
                        "Stone",
                        "Detected_Color",
                        "Detected_Origin",
                        "Reformatted_issuedDate",
                        "Indication",
                        "oldHeat",
                        "Mogok",
                        "Detected_Cut",
                        "Detected_Shape",
                        "carat",
                        "length",
                        "width",
                        "height"
                    ]]
                    result_df = result_df.rename(columns={
                        "Detected_Color": "Color",
                        "Detected_Origin": "Origin",
                        "Reformatted_issuedDate": "issuedDate",
                        "Detected_Cut": "Cut",
                        "Detected_Shape": "Shape"
                    })

                    # Append the DataFrame to the list
                    df_list.append(result_df)

        # Concatenate all DataFrames into one large DataFrame
        final_df = pd.concat(df_list, ignore_index=True)

        # Display the final DataFrame
        st.write(final_df)

        csv_data = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="Cert.csv",
            key="download-button"
        )

