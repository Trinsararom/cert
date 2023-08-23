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

# Define the OCR class
class OCR:

    def __init__(self):
        # Define cropping coordinates for different regions
        self.coordinates = [
            (570, 910, 168, 1010),
            (550, 980, 1800, 3000),
            (1580, 2180, 175, 1400)
        ]

    def process_cropped_images(self, img, coordinates):
        result = []

        for idx, (upper, lower, left, right) in enumerate(coordinates):
            crop = img[upper:lower, left:right]

            # Perform OCR on the cropped image
            extracted_text = pytesseract.image_to_string(crop)

            result.append(extracted_text.strip())

        return result

    def process_cropped_images1(self, img, coordinates):
        # Process cropped images and get the extracted text
        extracted_texts = self.process_cropped_images(img, coordinates)

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

    def extract_origin_info(self, image_path, coordinates):

        # Process cropped images and get the extracted text
        extracted_texts = self.process_cropped_images(image_path, coordinates)

        # Extract "Origin" information
        origin_info = None
        for text in extracted_texts:
            if 'Origin' in text:
                origin_info = text.split(':', 1)[1].strip()

        # Create a DataFrame with "Origin" as a column header and the extracted data as data
        data = {'Origin': [origin_info]}
        df = pd.DataFrame(data)

        return df

    def extrace_img3(self, img, coordinates):
        # Process the cropped image and extract text
        extracted_texts = self.process_cropped_images(img, coordinates)

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

    def detect_color(self, text):
        if "PigeonsBlood" in text:
            return "PigeonsBlood"
        elif "*" in text:
            return "PigeonsBlood"
        elif "pigeon-blood" in text:
            return "PigeonsBlood"
        else:
            return None

    def detect_cut(self, cut):
        if cut != "Cabochon":
            return "cut"
        else:
            return cut

    def detect_shape(self, shape):
        valid_shapes = [
            "cushion", "heart", "marquise", "octagonal", "oval",
            "pear", "rectangular", "round", "square", "triangular",
            "star", "sugarloaf", "tumbled"
        ]
        if shape in valid_shapes:
            return shape
        else:
            return "Others"

    def detect_origin(self, origin):
        if not origin.strip():
            return "No origin"

        # Remove words in parentheses
        origin_without_parentheses = re.sub(r'\([^)]*\)', '', origin)
        return origin_without_parentheses.strip()

    def reformat_issued_date(self, issued_date):
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

    def detect_mogok(self, origin):
        return str("(Mogok, Myanmar)" in origin)

    def generate_indication(self, comment):
        if comment in ["H", "H(a)", "H(b)", "H(c)"]:
            return "Heated"
        elif comment == "No indication of thermal treatment":
            return "Unheated"
        else:
            return "Unknown"

    def detect_old_heat(self, comment, indication):
        if indication == "Heated":
            if comment in ["H", "H(a)", "H(b)", "H(c)"]:
                return "oldHeat"
        return "Others"

    def generate_display_name(self, color, origin):
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
    def extract_cert_info(self, certNO):
        parts = certNO.split("-")
        if len(parts) == 2:
            cert_name = parts[0][:-4]  # Extract the first part of the certificate name (e.g., "GRS")
            cert_number = parts[0][-4:] + "-" + parts[1]  # Extract the last four characters and combine with the year
            return cert_name, cert_number
        return "", ""

    def convert_carat_to_numeric(self, value_with_unit):
        value_without_unit = value_with_unit.replace(" ct", "")
        numeric_value = float(value_without_unit)
        return numeric_value

    def convert_dimension(self, dimension_str):
        parts = dimension_str.split(" x ")
        if len(parts) == 3 and parts[-1].endswith(" (mm)"):
            length = float(parts[0])
            width = float(parts[1])
            height = float(parts[2][:-5])  # Remove " (mm)" from the last part
            return length, width, height
        return None, None, None

    def rename_identification_to_stone(self, dataframe):
        dataframe.rename(columns={"Identification": "Stone"}, inplace=True)
        return dataframe

    # Define the function to perform all data processing steps
    def perform_data_processing(self, result_df):

        result_df["Detected_Color"] = result_df["Color"].apply(self.detect_color)
        result_df["Detected_Cut"] = result_df["Cut"].apply(self.detect_cut)
        result_df["Detected_Shape"] = result_df["Shape"].apply(self.detect_shape)
        result_df["Detected_Origin"] = result_df["Origin"].apply(self.detect_origin)
        result_df["Reformatted_issuedDate"] = result_df["Date"].apply(self.reformat_issued_date)
        result_df["Mogok"] = result_df["Origin"].apply(self.detect_mogok)
        result_df["Indication"] = result_df["Comment"].apply(self.generate_indication)
        result_df["oldHeat"] = result_df.apply(lambda row: self.detect_old_heat(row["Comment"], row["Indication"]), axis=1)
        result_df["displayName"] = result_df.apply(lambda row: self.generate_display_name(row["Color"], row["Origin"]), axis=1)
        result_df["certName"], result_df["certNO"] = zip(*result_df["No."].apply(self.extract_cert_info))
        result_df["carat"] = result_df["Weight"].apply(self.convert_carat_to_numeric)
        result_df[["length", "width", "height"]] = result_df["Dimensions"].apply(self.convert_dimension).apply(pd.Series)
        result_df = self.rename_identification_to_stone(result_df)

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

# Create a Streamlit App
def main():
    st.set_page_config(
        page_title="Cert",
        layout='wide',
    )

    st.title('Certificate Scraper')

    # Create an instance of the OCR class
    ocr = OCR()

    # Create a Streamlit file uploader for the zip file
    zip_file = st.file_uploader("Upload a ZIP file containing images", type=["zip"])

    if zip_file is not None:
        # Extract the uploaded ZIP file
        with zipfile.ZipFile(zip_file) as zip_data:
            df_list = []

            for image_file in zip_data.namelist():
                # Rest of your code for processing images and creating DataFrames goes here...
                pass

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

# Run the Streamlit app
if __name__ == "__main__":
    main()

