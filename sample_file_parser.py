import zipfile
import os
import pandas as pd
import numpy as np
import sample
import math
from csv import reader
import constants

#CSV file header columns names
brand_col_name = "Brand"
product_col_name = "Product"
device_col_name = "Device"
card_col_name = "Card"
note_col_name = "Note"
#CSV file data columns names
id_col_name = "sample"
tags_col_name = "tags"
card_num_col_name = "card"
default_device = "prototype_1_aromabit"

Samples = []

#extract sample files from a zip file and creates a list of samples
def extract_zip_file(file_path):
    try:
        with zipfile.ZipFile(file_path, "r") as f:
            for name in f.namelist():
                file = f.open(name)
                header_df = pd.read_csv(file, nrows=1)
                file.close()
                file = f.open(name)
                df = pd.read_csv(file, skiprows=2)
                df._path = name
                nextsample = detect_sample_attr(header_df)
                nextsample.ID = df[id_col_name][0]
                nextsample.tags = df[tags_col_name][0]
                nextsample.values = df.drop(columns=[id_col_name, tags_col_name, card_num_col_name])
                Samples.append(nextsample)
    except Exception as e:
        print("Open zip error", e)
        return None

    return Samples
# axtracts the samples general attributes from the attribute table
def detect_sample_attr(attr):
    attr.columns = attr.columns.str.strip()
    
    nextsample = sample.DataSample()
    nextsample.brand = attr[brand_col_name][0]
    if type(nextsample.brand) == str:
        nextsample.brand.strip()
    nextsample.product = attr[product_col_name][0]
    if type(nextsample.product) == str:
        nextsample.product.strip()    
    nextsample.sampler_type = attr[device_col_name][0]
    if type(nextsample.sampler_type) == str:
        nextsample.sampler_type.strip()    
    elif np.isnan(nextsample.sampler_type):
        nextsample.sampler_type = default_device
    nextsample.card = attr[card_col_name][0]
    if type(nextsample.card) == str:
        nextsample.card.strip()    
    nextsample.note = attr[note_col_name][0]
    if type(nextsample.note) == str:
        nextsample.note.strip()    
    
    return nextsample


def get_sample_prod(sample_id):
    prod = ""
    for samp in Samples:
        if sample_id == samp.ID:
            prod = samp.product
    return prod

def get_sample_tag(sample_id):
    tags = ''
    for samp in Samples:
        if sample_id == samp.ID:
            tags = samp.tags
    return tags    

def get_sample_card(sample_id):
    card = ''
    for samp in Samples:
        if sample_id == samp.ID:
            card = samp.card
    return card 