import glob
import gc

import numpy as np
import pd as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import requests
import gzip
from io import BytesIO
from datetime import datetime
from tqdm import tqdm
from selenium.webdriver.chrome.service import Service
import re

url = "https://placedata.reddit.com/data/canvas-history/index.html"


def getUrl(web_url):
    '''
    this method extract the url from a website.
    :param web_url: the website url.
    :return: list of urls.
    '''
    # Launch a new instance of the Chrome browser
    chrome_options = webdriver.ChromeOptions()
    service = Service("C://webdrivers//chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    # Navigate to the URL of the web page
    driver.get(web_url)
    # Find all the <a> tags in the web page and extract their "href" attributes
    urls = []
    for link in driver.find_elements(By.TAG_NAME, "a"):
        url = link.get_attribute("href")
        if url is not None:
            urls.append(url)
    driver.quit()
    return urls


def createDF(url):
    '''
    :param url:
    :return: dataframe with the data inside the url.
    '''
    response = requests.get(url)
    compressed_file = BytesIO(response.content)
    try:
        decompressed_file = gzip.GzipFile(fileobj=compressed_file)

        # Read and process the data in chunks
        chunk_size = 1000  # Adjust the chunk size as per your requirements
        dfs = []
        for chunk in pd.read_csv(
                decompressed_file,
                header=0,
                names=["timestamp", "user_id", "pixel_color", "coordinate"],
                chunksize=chunk_size
        ):
            # Process and filter the data as needed
            chunk['timestamp'] = chunk['timestamp'].str[:-4]
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format="ISO8601")
            temp_max_timestamp = "2022-04-04 22:47:40.185000"
            iso8601_format = "%Y-%m-%d %H:%M:%S.%f"
            max_timestamp = datetime.strptime(temp_max_timestamp, iso8601_format)
            filtered_df = chunk[chunk['timestamp'] <= max_timestamp]

            # Split coordinate string directly into separate x and y columns
            # filtered_df[['x', 'y']] = filtered_df['coordinate'].str.split(',', expand=True)
            filtered_df['x'] = filtered_df['coordinate'].str.split(',').str[0]
            filtered_df['y'] = filtered_df['coordinate'].str.split(',').str[1]

            # Convert x and y columns to numeric type if desired
            filtered_df['x'] = pd.to_numeric(filtered_df['x'])
            filtered_df['y'] = pd.to_numeric(filtered_df['y'])

            # Drop the original "coordinate" column
            filtered_df.drop('coordinate', axis=1, inplace=True)

            dfs.append(filtered_df)

        return pd.concat(dfs)
    except Exception as e:
        print(f"Error reading file: {url}")
        print(e)
        return None


# urls = getUrl(url)
# for i in urls[:2]:
#     df = createDF(i)


def createDF2Rows(url, chunk_size=1000):
    '''

    :param url:
    :param chunk_size:
    :return: first 2 rows of the dataframe with the data inside the url.
    '''
    response = requests.get(url)
    compressed_file = BytesIO(response.content)
    try:
        decompressed_file = gzip.GzipFile(fileobj=compressed_file)
        df_chunks = pd.read_csv(decompressed_file, header=0,
                                names=["timestamp", "user_id", "pixel_color", "coordinate"], chunksize=chunk_size)
        for df_chunk in df_chunks:
            return df_chunk[:2]
    except Exception as e:
        print(f"Error reading file: {url}")
        print(e)
        return None


#

# def getLastTimestamp(urls):
#     '''
#
#     :param urls: list of urls
#     :return: the last timestamp before all the pixels turned white at the end of the experiment.
#     '''
#     smallest_timestamp = datetime(1, 1, 1, 0, 0, 0).isoformat() + 'Z'
#     iso8601_format = "%Y-%m-%dT%H:%M:%SZ"
#     final_max_timestamp = datetime.strptime(smallest_timestamp, iso8601_format)
#     conn = sqlite3.connect('database.db')
#     cursor = conn.cursor()
#     cursor.execute('''CREATE TABLE IF NOT EXISTS max_timestamps (
#                         id INTEGER PRIMARY KEY AUTOINCREMENT,
#                         max_timestamp TEXT)''')
#     conn.commit()
#
#     batch_size = 10
#     url_batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
#     for batch in tqdm(url_batches):
#         for url in batch:
#             # Download the gzipped csv file from the URL
#             df = createDF(url)
#             df['timestamp'] = df['timestamp'].str[:-4]
#             df['timestamp'] = pd.to_datetime(df['timestamp'], format="ISO8601")
#             # find the time everyone changed to white
#             temp_max = df.loc[df['pixel_color'] != '#FFFFFF', 'timestamp'].max()
#             if pd.isna(temp_max):
#                 continue
#             df = df.dropna(subset=['timestamp'])
#             temp_max = pd.Timestamp(temp_max)
#             if temp_max > final_max_timestamp:
#                 final_max_timestamp = temp_max
#             del df
#             gc.collect()
#             cursor.execute("INSERT INTO max_timestamps (max_timestamp) VALUES (?)", (str(final_max_timestamp),))
#             conn.commit()
#     conn.close()
#
#     conn = sqlite3.connect('database.db')
#     cursor = conn.cursor()
#     cursor.execute("SELECT max(max_timestamp) FROM max_timestamps")
#     results = cursor.fetchall()
#     conn.close()
#     for row in results:
#         return row[0]


#

def lastColor(pixel_df):
    '''

    :param pixel_df: dataframe
    :return: a dataframe with the columns coordinate, pixel_color and timestamp, where values returned matched with the
     last data that happened before all pixels turned white.
    '''
    max_idx = pixel_df.groupby(['x', 'y'])['timestamp'].idxmax()
    result = pixel_df.loc[max_idx, ['x', 'y', 'pixel_color', 'timestamp']]
    return result


def placementUpdate(pixel_df):
    '''

    :param pixel_df: dataframe
    :return: same daaframe with the addition of the column pixel_allocations that sum the amount of allocations
    per pixel.
    '''
    coord_count = pixel_df.groupby(['x', 'y']).size().reset_index(name="pixel_allocations")
    return coord_count


#
# urls = getUrl(url)
# sum_of_allocations = 0
# temp_max_timestamp = "2022-04-04 22:47:40.185000"
# iso8601_format = "%Y-%m-%d %H:%M:%S"
# max_timestamp = datetime.strptime(temp_max_timestamp, iso8601_format)
# for url in tqdm(urls):
#     temp_df = createDF(url)
#     if temp_df is not None:
#         sum_of_allocations += temp_df.shape[0]
#         del temp_df
#         gc.collect()
# print(sum_of_allocations)


def extract_first_timestamp(url):
    '''

    :param url:
    :return: the timestamp the data inside the url started from.
    '''
    df = createDF2Rows(url)
    while df is None:
        continue
    if df is not None and len(df) > 0:
        df['timestamp'] = df['timestamp'].str[:-4]
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="ISO8601")
        first_timestamp = df['timestamp'].iloc[0]  # Extract the first timestamp value
        return first_timestamp


def myFunc(urls, batch_size=4):
    '''
    :param urls: list of 80 urls from the website of reddit.
    :param batch_size:
    :return: 16 csv files of dataframes with the columns : coordinate,pixel_allocations,final_color and timestamp.
    '''
    url_batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
    result_files = []
    sum_of_allocations = 0
    for batch_idx, url_batch in enumerate(tqdm(url_batches)):
        temp_pixel_allocations = pd.DataFrame(columns=['x', 'y', 'pixel_allocations', 'timestamp'])
        temp_color_data = pd.DataFrame(columns=['x', 'y', 'final_color', 'timestamp'])
        record_covered = 0
        for url_idx, url in enumerate(url_batch):
            try:
                df = createDF(url)
                if df is None:
                    continue
                temp_placement_counter = placementUpdate(df)
                temp_pixel_allocations = pd.concat([temp_pixel_allocations, temp_placement_counter])
                color_data = lastColor(df)
                color_data.rename(columns={'pixel_color': 'final_color'}, inplace=True)
                temp_color_data = pd.concat([temp_color_data, color_data])
                assert temp_pixel_allocations['pixel_allocations'].sum() == df.shape[0] + record_covered
                record_covered += df.shape[0]
                del df, temp_placement_counter, color_data
                gc.collect()
            except Exception as e:
                print(f"Error processing URL: {url}")
                print(e)

        temp_pixel_allocations = temp_pixel_allocations.groupby(['x', 'y']).agg(
            {'pixel_allocations': 'sum'}).reset_index()
        max_timestamps = temp_color_data.groupby(['x', 'y'])['timestamp'].max().reset_index()
        temp_color_data = temp_color_data.merge(max_timestamps, on=['x', 'y', 'timestamp'])
        temp_color_data.drop_duplicates(subset=['x', 'y'], keep='first', inplace=True)
        merged_df = temp_pixel_allocations.merge(temp_color_data, on=['x', 'y'], suffixes=('', ''))
        temp_sum = merged_df['pixel_allocations'].sum()
        if temp_sum == record_covered:
            result_filename = f"file_batch_{batch_idx}.csv"
            result_files.append(result_filename)
            merged_df.to_csv(result_filename, index=False)
            del temp_pixel_allocations, temp_color_data, merged_df
            gc.collect()
            sum_of_allocations += record_covered
            print(f'total pixel_allocations until batch number {batch_idx} is: ' + str(sum_of_allocations))
        else:
            print(f'error in num of allocations in batch {batch_idx}!, got {temp_sum} instead of {record_covered}')


# urls = getUrl(url)
# myFunc(urls)


# result_files = glob.glob('file*.csv')
# result =0
# for file in result_files:
#     df = pd.read_csv(file)
#     result += df['pixel_allocations'].sum()
# print(result)


'TESTING THE RESULT'


def merge_2df(file1, file2):
    # first_num = re.findall(r'\d+\+\d+', file1)[0]
    # second_num = re.findall(r'\d+\+\d+', file2)[0]

    # Read the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Merge the two dataframes based on 'x' and 'y' columns
    merged_df = pd.merge(df1, df2, on=['x', 'y'], how='outer')
    merged_df['pixel_allocations_x'].fillna(0, inplace=True)
    merged_df['pixel_allocations_y'].fillna(0, inplace=True)

    # Sum the 'pixel_allocations' column
    merged_df['pixel_allocations'] = merged_df['pixel_allocations_x'] + merged_df['pixel_allocations_y']

    # Convert timestamps to datetime objects
    merged_df['timestamp_x'] = pd.to_datetime(merged_df['timestamp_x'])
    merged_df['timestamp_y'] = pd.to_datetime(merged_df['timestamp_y'])
    merged_df['pixel_allocations'] = merged_df['pixel_allocations'].astype(int)

    # Take the maximum timestamp between the matching rows
    merged_df['timestamp'] = merged_df[['timestamp_x', 'timestamp_y']].max(axis=1)
    merged_df['final_color'] = np.where(
        merged_df['final_color_x'].notnull() & merged_df['final_color_y'].notnull(),
        np.where(merged_df['timestamp_y'] > merged_df['timestamp_x'], merged_df['final_color_y'],
                 merged_df['final_color_x']),
        np.where(merged_df['final_color_x'].isnull(), merged_df['final_color_y'], merged_df['final_color_x']))

    # Select the final columns for the result
    final_result = merged_df[['x', 'y', 'final_color', 'timestamp', 'pixel_allocations']]

    # Save the final result to CSV
    final_result.to_csv(f'final_df.csv', index=False)


# Get a list of file_batch CSV files
result = 0
file_list = glob.glob('final_df.csv')
for file in file_list:
    df = pd.read_csv(file)
    result += df['pixel_allocations'].sum()
print(result)

# Iterate over the file list and merge two files at a time
# for  i in tqdm(range(0, len(file_list), 2)):
#     file1 = file_list[i]
#     file2 = file_list[i + 1]
#     merge_2df(file1, file2)
    # print(file1,file2,num)


#
# result_files = glob.glob('result*.csv')
# sum = 0
# for file in tqdm(result_files):
#     df = pd.read_csv(file)
#     sum += df['pixel_allocations'].sum()
# print(sum)


#
#
# for i in tqdm(range(0, len(result_files), 2)):
#     file1 = result_files[i]
#     file2 = result_files[i + 1]
#     merge_2df(file1, file2)


