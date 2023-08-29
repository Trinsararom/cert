import pytesseract
from PIL import Image
import cv2
from datetime import datetime
import os
import re
import zipfile
import io
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Cert",
    layout = 'wide',
)

st.title('Certicatate Scraper')

# Initialize the Tesseract OCR
def initialize_tesseract():
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Initialize Tesseract
initialize_tesseract()

def crop_image(img):

    # Get the dimensions of the original image
    height, width, channels = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)

    # Calculate the width for crop1 (e.g., 50% of the original width)
    crop1_width = width // 2

    # Calculate the width for crop2
    crop2_width = width - crop1_width

    # Calculate the coordinates for cropping crop1 and crop2
    top = 0
    bottom = height
    left_crop1 = 0
    right_crop1 = crop1_width
    left_crop2 = crop1_width
    right_crop2 = width

    # Crop the two parts
    crop1 = img[top:bottom, left_crop1:right_crop1]
    crop2 = img[top:bottom, left_crop2:right_crop2]

    # Calculate the dimensions for crop3 (top half of crop2)
    crop3_height = int(crop2.shape[0] // 2.4)

    # Crop the top half of crop2 to create crop3
    top_crop3 = 0
    bottom_crop3 = crop3_height
    left_crop3 = 0
    right_crop3 = crop2.shape[1]

    crop3 = crop2[top_crop3:bottom_crop3, left_crop3:right_crop3]

    return crop1, crop3

def process_cropped_images(img):
    # Perform OCR on the cropped image
    extracted_text = pytesseract.image_to_string(img)

    return extracted_text

def extract_gemstone_info(img):
    # Assuming you have the functions crop_image and process_cropped_images defined elsewhere
    crop1, crop3 = crop_image(img)

    # Process cropped images and get the extracted text
    extracted_texts = process_cropped_images(crop1)

    extracted_texts = extracted_texts.replace('‘', "")
    # Split the text into lines
    lines = extracted_texts.split('\n')

    # Initialize variables to store extracted information
    extracted_info = {}

    # Keywords in lowercase
    keywords = ["no. grs", "date", "object", "identification", "weight", "dimensions", "cut", "shape", "color", "comment"]

    # Iterate through the lines to find relevant information
    for line in lines:
        line_lower = line.lower()
        for keyword in keywords:
            if line_lower.startswith(keyword):
                key, value = line.split(maxsplit=1)
                extracted_info[keyword] = value.strip()

    # Define custom column names
    custom_column_names = ["No.", "Date", "Object", "Identification", "Weight", "Dimensions", "Cut", "Shape", "Color", "Comment"]

    # Create a DataFrame from the extracted information using custom column names
    data_dict = {}
    for keyword, col_name in zip(keywords, custom_column_names):
        value = extracted_info.get(keyword, "")
        if keyword == "no. grs":
            value = lines[5].strip() if keyword not in extracted_info else extracted_info[keyword]
        elif keyword == "date":
            value = lines[6].strip() if keyword not in extracted_info else extracted_info[keyword]
        elif keyword == "object":
            value = lines[7].strip() if keyword not in extracted_info else extracted_info[keyword]
        elif keyword == "identification":
            value = lines[8].strip() if keyword not in extracted_info else extracted_info[keyword]
        data_dict[col_name] = [value]

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data_dict)

    return df

def extract_origin_info(img):
    # Crop the image
    crop1, crop3 = crop_image(img)

    # Process cropped images and get the extracted text
    extracted_texts = process_cropped_images(crop3)

    # Split the extracted text into lines and filter out empty lines
    lines = [line for line in extracted_texts.splitlines() if line.strip()]

    # Create a DataFrame
    df = pd.DataFrame({'Origin': [lines[-1]]})
    
    return df


def detect_color(text):
    if "Color is PigeonsBlood" in text:
        return "PigeonsBlood"
    elif "Color contains *" in text:
        return "PigeonsBlood"
    elif "(GRS type \"pigeon's blood\")"  in text:
        return "PigeonsBlood"
    else:
        return text
    
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
    origin_without_parentheses = origin
    return origin_without_parentheses.strip()

def reformat_issued_date(issued_date):

    # Remove ordinal suffixes (e.g., "th", "nd", "rd")
    cleaned_date = re.sub(r'(?<=\d)(st|nd|rd|th)\b', '', issued_date.replace("‘", "").replace("I", "1").replace("S", "5").strip())

    # Parse the cleaned date string
    #parsed_date = datetime.strptime(cleaned_date, '%d %B %Y')

    # Reformat the date to YYYY-MM-DD
    reformatted_date = cleaned_date
    return reformatted_date

    
def detect_mogok(origin):
    return str("(Mogok, Myanmar)" in origin)

def generate_indication(comment):
    if comment in ["H", "H(a)", "H(b)", "H(c)"]:
        return "Heated"
    else:
        return "Unheated"
    
def detect_old_heat(comment, indication):
    if indication == "Heated":
        if comment in ["H", "H(a)", "H(b)", "H(c)"]:
            return "oldHeat"
    return "Others"

def generate_display_name(color, origin):
    display_name = ""

    if color is not None:
        if "*" in color:
            display_name = "PGB*"
        elif color == "red":
            display_name = "GRS"
        elif color == "PigeonsBlood":
            display_name = "PGB"
    
    if "(Mogok, Myanmar)" in origin:
        display_name = "MG-" + display_name
    
    return display_name

# Define the function to extract the year and number from certNO
def extract_cert_info(df,certNO):
    # Split the specified column into two columns
    df[['certName', 'certNO']] = df[certNO].str.extract(r'(\D+)(\d+.*)')
    return df

def convert_carat_to_numeric(value_with_unit):
    value_without_unit = value_with_unit.replace(" ct", "").replace(" et", "").replace(" ot", "")
    numeric_value = (value_without_unit)
    return numeric_value

def convert_dimension(dimension_str):
    parts = dimension_str.replace("—_", "").replace("_", "").replace("§", "5").replace(",", ".").replace("=", "").split(" x ")
    if len(parts) == 3 and parts[-1].endswith(" (mm)"):
        length = (parts[0])
        width = (parts[1])
        height = (parts[2][:-5])  # Remove " (mm)" from the last part
        return length, width, height
    return None, None, None

def rename_identification_to_stone(dataframe):
    dataframe.rename(columns={"Identification": "Stone"}, inplace = True)
    dataframe["Stone"] = dataframe["Stone"].str.replace("‘", "").str.strip()
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
    result_df["displayName"] = result_df.apply(lambda row: generate_display_name(row["Detected_Color"], row["Detected_Origin"]), axis=1)
    result_df = extract_cert_info(result_df, 'No.')
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
                    # Process the image and perform data processing
                    df_1 = extract_gemstone_info(img)
                    df_2 = extract_origin_info(img)
                    result_df = pd.concat([df_1, df_2], axis=1)
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
