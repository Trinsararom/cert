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

st.title('Cert Scraper')

# Initialize the Tesseract OCR
def initialize_tesseract():
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Initialize Tesseract
initialize_tesseract()


def crop_image(img):

    
    # Get the dimensions of the original image
    height, width, channels = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)

    crop1_width = width // 2

    # Calculate the width for crop2
    crop2_width = width - crop1_width

    # Calculate the coordinates for cropping crop1, crop2, and crop3
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

    crop4_height = int(crop1.shape[0] // 2.4)

    # Crop the top half of crop2 to create crop3
    top_crop4 = int(crop1.shape[0] // 4.3)
    bottom_crop4 = crop4_height
    left_crop4 = 0
    right_crop4 = crop1.shape[1]

    crop4 = crop1[top_crop4:bottom_crop4, left_crop4:right_crop4]



    # Crop the top half of crop2 to create crop3
    top_crop5 = int(crop1.shape[0] // 1.6)
    bottom_crop5 = int(crop1.shape[0] // 1.13)
    left_crop5 = int(crop1.shape[1] // 19.2444)
    right_crop5 = crop1.shape[1]

    crop5 = crop1[top_crop5:bottom_crop5, left_crop5:right_crop5]

    return crop1, crop3, crop4, crop5

def process_cropped_images(img):
    # Perform OCR on the cropped image
    extracted_text = pytesseract.image_to_string(img)

    return extracted_text

def extract_gemstone_info(img):
    # Assuming you have the functions crop_image and process_cropped_images defined elsewhere
    crop1, crop3, crop4, crop5 = crop_image(img)
    try:
        # Process cropped images and get the extracted text
        extracted_texts = process_cropped_images(crop1)

        extracted_texts = extracted_texts.replace('‘', "").replace(',', ".")
        # Split the text into lines
        lines = extracted_texts.split('\n')

        # Initialize variables to store extracted information
        extracted_info = {}

        # Keywords in lowercase
        keywords = ["no", "date", "object", "identification", "weight", "dimensions", "cut", "shape", "color", "comment"]

        # Iterate through the lines to find relevant information
        for line in lines:
            line_lower = str(line).lower()
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
            if keyword == "no grs":
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

    except ValueError:

        # Process cropped images and get extracted text for crop4
        extracted_texts4 = process_cropped_images(crop4)
        extracted_texts4 = extracted_texts4.replace('‘', "").replace(',', ".")
        lines4 = extracted_texts4.split('\n')
        lines4 = [line for line in lines4 if line.strip() != ""]

        # Create a DataFrame from the extracted information in extracted_texts4
        data_dict4 = {line.split()[0]: [line.split(maxsplit=1)[1]] for line in lines4}
        df4 = pd.DataFrame(data_dict4)
        df4.columns = ['No.', 'Date', 'Object', 'Identification']

        # Process cropped images and get extracted text for crop5
        extracted_texts5 = process_cropped_images(crop5)
        lines5 = extracted_texts5.split('\n')
        lines5 = [line for line in lines5 if line.strip() != ""]

        # Separate headers and values for crop5
        headers5 = lines5[:6]
        values5 = lines5[6:]

        # Create a dictionary from the headers and values for crop5
        data_dict5 = {headers5[i]: [values5[i]] for i in range(len(headers5))}
        df5 = pd.DataFrame(data_dict5)
        df5.columns = ['Weight', 'Dimensions', 'Cut', 'Shape', 'Color', 'Comment']

        # Concatenate df and df1 vertically (row-wise)
        df = pd.concat([df4, df5], axis = 1)

        return df

def extract_origin_info(img):
    # Crop the image
    crop1, crop3, crop4, crop5 = crop_image(img)

    # Process cropped images and get the extracted text
    extracted_texts = process_cropped_images(crop3)

    # Split the extracted text into lines and filter out empty lines
    lines = [line for line in extracted_texts.splitlines() if line.strip()]
    lines = [line for line in lines if "Special comment see appendix" not in line]

        # Check if lines is empty, and return an empty DataFrame if it is
    if not lines:
        return pd.DataFrame({'Origin': [""]})
    # Create a DataFrame
    df = pd.DataFrame({'Origin': [lines[-1]]})

    return df


def detect_color(text):
    text = str(text).lower()  # Convert the text to lowercase
    if "pigeonsblood" in text:
        return "PigeonsBlood"
    elif "royal blue" in text or "(grs type \"royal blue\")" in text or "(gr type \"royal blue\")" in text:
        return "RoyalBlue"
    elif "*" in text:
        return "PigeonsBlood"
    elif "vibrant" in text or "(grs type \"vibrant\")" in text or "(gr type \"vibrant\")" in text :
        return "VibrantVividPink"
    elif "(grs type \"pigeon's blood\")"  in text:
        return "PigeonsBlood"
    elif "(gr type \"pigeon's blood\")" in text:
        return "PigeonsBlood"
    elif "pigeon's blood" in text:
        return "PigeonsBlood"
    elif "pigeons blood" in text :
        return "PigeonsBlood"
    else:
        return text
    
def detect_cut(cut):
    text = str(cut).lower()
    if "sugar loaf" in text :
        return "sugar loaf"
    elif "cabochon" in text:
        return "cabochon"
    else:
        return "cut"
        
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
    try:
        # Remove ordinal suffixes (e.g., "th", "nd", "rd")
        cleaned_date = re.sub(r'(?<=\d)(st|nd|rd|th)\b', '', issued_date.replace("‘", "").strip())

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
    if comment in ["H", "H(a)", "H(b)", "H(c)", "(a)", "Ha)", "H(a)", "(b)", "Hb)", "H(b)", "(c)", "Hc)", "H(c)"]:
        return "Heated"
    else:
        return "Unheated"

    
def detect_old_heat(comment, indication):
    if indication == "Heated":
        if "(a)" in comment or "Ha)" in comment or "H(a" in comment:
            return "H(a)"
        elif "(b)" in comment or "Hb)" in comment or "H(b" in comment:
            return "H(b)"
        elif "(c)" in comment or "Hc)" in comment or "H(c" in comment:
            return "H(c)"
        return comment
    else :
        comment = ''
        return comment
    
def generate_display_name(color, Color_1, origin, indication, comment):
    display_name = ""

    if color is not None:
        color = str(color).lower()  # Convert color to lowercase
        if indication == "Unheated":
            if "*" in color:
                display_name = "GRS(PGB*)"
            elif color == "pigeonsblood" or "(grs type \"pigeon's blood\")" in color or "pigeon's blood" in color or "(gr type \"pigeon's blood\")" in color:
                display_name = "GRS(PGB)"
            elif color == "royal blue" or "royalblue" in color or "royal blue" in color or "(grs type \"royal blue\")" in color or "(gr type \"royal blue\")" in color:
                display_name = "GRS(RYB)"
            else:
                display_name = f"GRS({Color_1})"
        if indication == "Heated": 
            if "(" in comment and ")" in comment:
                comment_match = re.search(r'\(([^)]+)\)', comment)
                if comment_match:
                    comment = comment_match.group(1)
            else:
                comment = comment
            if "*" in color:
                display_name = "GRS(PGB*)"
            elif color == "pigeonsblood" or "(grs type \"pigeon's blood\")" in color or "pigeon's blood" in color or "(gr type \"pigeon's blood\")" in color:
                display_name = f"GRS(PGB)({comment})"
            elif color == "royal blue" or "royalblue" in color or "royal blue" in color or "(grs type \"royal blue\")" in color or "(gr type \"royal blue\")" in color:
                display_name = f"RYB({comment})"
            else:
                display_name = f"GRS({Color_1})({comment})"
    
    if "(mogok, myanmar)" in str(origin).lower():  # Convert origin to lowercase for case-insensitive comparison
        display_name = "MG-" + display_name
    
    return display_name


# Define the function to extract the year and number from certNO
def extract_cert_info(df,certName):
    # Split the specified column into two columns
    df['certName'] = df[certName].str.extract(r'([A-Z]+)')
    df['certNO'] = df[certName]
    return df

def convert_carat_to_numeric(value_with_unit):
    value_without_unit = value_with_unit.replace(" ct", "").replace(" et", "").replace(" ot", "").replace("ct", "")
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
    # Rename "Identification" to "Stone"
    dataframe.rename(columns={"Identification": "Stone"}, inplace=True)
    # Remove unwanted words and trim spaces in the "Stone" column
    dataframe["Stone"] = dataframe["Stone"].str.replace("‘", "").str.strip()

    # Define a list of gemstone names to detect
    gemstone_names = ["Ruby", "Emerald", "Pink Sapphire", "Purple Sapphire", "Sapphire", "Spinel", "Tsavorite", "Blue Sapphire", "Fancy Sapphire", "Peridot", "Padparadscha"]  # Add more gemstone names as needed

    # Function to remove "Natural" or "Star" from the stone name
    def remove_prefix(name):
        for prefix in ["Natural", "Star"]:
            name = name.replace(prefix, "").strip()
        return name

    # Detect and update the "Stone" column with the gemstone names (ignoring "Natural" or "Star")
    dataframe["Stone"] = dataframe["Stone"].apply(lambda x: next((gem for gem in gemstone_names if gem in remove_prefix(x)), x))

    return dataframe

def detect_vibrant(Vibrant):
    Vibrant = str(Vibrant).lower() 
    return str("vibrant" in Vibrant)
    
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
    result_df["displayName"] = result_df.apply(lambda row: generate_display_name(row["Color"], row['Detected_Color'], row["Detected_Origin"], row['Indication'], row['oldHeat']), axis=1)
    result_df = extract_cert_info(result_df, 'No.')
    result_df["carat"] = result_df["Weight"].apply(convert_carat_to_numeric)
    result_df[["length", "width", "height"]] = result_df["Dimensions"].apply(convert_dimension).apply(pd.Series)
    result_df['Detected_Origin'] = result_df['Detected_Origin'].str.replace(r'\(.*\)', '').str.strip()
    result_df[['carat', 'length', 'width', 'height']] = result_df[['carat', 'length', 'width', 'height']].replace("$", "5").replace("| ", "1")
    result_df = rename_identification_to_stone(result_df)
    result_df['Vibrant'] = result_df["Detected_Color"].apply(detect_vibrant)

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
    "Vibrant",
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
                try:
                    # Read the image
                    with zip_data.open(image_file) as file:
                        img_data = io.BytesIO(file.read())
                        img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 0)
                        noiseless_image_bw = cv2.fastNlMeansDenoising(img, None, 20, 7, 21)
                        img = noiseless_image_bw
                        
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
                            "Vibrant",
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
                except Exception as e:
                    # Handle errors for this image, you can log or print the error message
                    st.error(f"Error processing image {image_file}: {str(e)}")
                    pass  # Skip to the next image

        # Concatenate all DataFrames into one large DataFrame
        final_df = pd.concat(df_list, ignore_index=True)

        # Display the final DataFrame
        st.write(final_df)


        csv_data = final_df.to_csv(index=False, float_format="%.2f").encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="Cert.csv",
            key="download-button"
        )
