import random
from apo4rec.util import generate_responses_and_extract_outputs, generate_responses, extract_summary_1, extract_summary_2
from apo4rec.data import subsample_data
import numpy as np

from time import perf_counter



class Filter:
    def __init__(self,
                 data_path):
        self.data_path = data_path
        self.title2detail = self.load_title2detail()
        self.simMatrix = []
        if 'ml1m' in data_path:
            self.key_attribute_types = ['genre', 'actor', 'director', 'tag']
        elif 'cd' in data_path:
            self.key_attribute_types = ['category', 'artist']
        elif 'game' in data_path:
            self.key_attribute_types = ['category', 'company']
        elif 'kindle' in data_path:
            self.key_attribute_types = ['category', 'author']

    def load_title2detail(self):
        title2detail = {}
        file_path = self.data_path + '/i_idx2str.dat'
        line_count = 0
        attribute_types = []
        with open(file_path) as fin:
            for line in fin:
                if line_count == 0:
                    # itemIdx:::title:::year:::genre:::director:::actor:::tag:::country
                    attribute_types = line.strip().split(':::')
                else:
                    records = line.strip().split(':::')
                    # itemIdx = int(records[0])
                    title = records[1].strip()
                    title2detail[title] = {}
                    for i, record in enumerate(records):
                        record = record.replace('[\'', '').replace('\']', '')
                        record_list = record.strip().split('\', \'')
                        attribute_type = attribute_types[i]
                        title2detail[title][attribute_type] = record_list
                line_count += 1
        return title2detail

    def check_mismatches(self, input_item, other_items, matrix):
        bool_vec = []
        for item in other_items:
            bool_vec.append(self.check_mismatch(input_item, item, matrix))
        bool_vec = np.array(bool_vec)
        return bool_vec.sum()

    def check_mismatch(self, input_item_title, target_item_title):

        num_common = 0
        for attribute_type in self.key_attribute_types:
            input_attribute_values = set(self.title2detail[input_item_title][attribute_type])
            target_attribute_values = set(self.title2detail[target_item_title][attribute_type])
            num_common += len(input_attribute_values & target_attribute_values)

        return num_common

    def check_match(self, input_item_title, target_item_title):
        num_common = 0
        for attribute_type in self.key_attribute_types:
            input_attribute_values = set(self.title2detail[input_item_title][attribute_type])
            target_attribute_values = set(self.title2detail[target_item_title][attribute_type])
            num_common += len(input_attribute_values & target_attribute_values)
        if num_common >= 4:
            return True
        else:
            return False



