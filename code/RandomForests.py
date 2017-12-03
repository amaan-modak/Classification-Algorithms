import copy
import math
import statistics
from random import randint

columns = []

class TreeNode:
    def __init__(self):
        self.label = None
        self.branches = []
        self.attribute_index = None
        self.attribute_value = None

    def add_child(self, obj):
        self.branches.append(obj)

class DataRow:
    def __init__(self, truth, data):
        self.truth = truth
        self.data = data
        self.weight = None

class Column:
    def __init__(self, data):
        self.choices = list(set(data))

def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False        
        
def read_file():
    input_file = 'project3_dataset2.txt' #Change this for changing input file
    array = []

    with open(input_file) as file:
        for row in file:
            data = row.split('\t')
            data = [float(x) if is_number(x) else x for x in data]
            #print(data)
            data_row = DataRow(data[-1], data[:-1])   
            #print(data_row.data)
            array.append(data_row)
    #print(len(array))
    return array

def split_data(data, split_value):
    training_set = math.floor(split_value * len(data))
    return data[:training_set], data[training_set:]

def normalize_data(inputs):
    #print(inputs[0].data)
    for j in range(len(inputs[0].data)):
        if type(inputs[0].data[j]) is not str:
            min_elem, max_elem = None, None
            for i in range(len(inputs)):
                if min_elem is None:
                    min_elem = min(list(zip(*[x.data for x in inputs]))[j])
                    max_elem = max(list(zip(*[x.data for x in inputs]))[j])
                inputs[i].data[j] = (inputs[i].data[j] - min_elem) / (max_elem - min_elem)
                #print(inputs[i].data[j])
                
def entropy(inputs):
    positives = inputs.count(1.0)
    negatives = len(inputs) - positives
    total = len(inputs)

    if positives == 0 or negatives == 0:
        return 0
    else:
        entropy = -(positives/total) * math.log(positives/total,2) - (negatives/total) * math.log(negatives/total,2)
        return entropy
    
def information_gain(input_data, attribute_index):
    subset_entropy = 0
    #print(columns[attribute_index].choices)
    for choice in columns[attribute_index].choices:
        input_data_subset = [x.truth for x in input_data if x.data[attribute_index] == choice]
        subset_entropy += (len(input_data_subset) / len(input_data)) * entropy(input_data_subset)
    input_data = [x.truth for x in input_data]
    res_entropy = entropy(input_data) - subset_entropy
    #print(columns[attribute_index].choices, res_entropy)
    return res_entropy

def best_index(input_data, attribute_list):
    input_data = input_data[:]
    best_gain = -1
    best_attr = -1

    for attribute_index in attribute_list:
        temp_gain = information_gain(input_data, attribute_index)
        if temp_gain > best_gain:
            best_gain = temp_gain
            best_attr = attribute_index
    return best_attr

def get_majority_label(input_data):
    try:
        return statistics.mode([x.truth for x in input_data])
    except statistics.StatisticsError:
        return [x.truth for x in input_data].pop()

def get_columns(inputs):
    for i in range(len(inputs[0].data)):
        columns.append(Column(list(zip(*[x.data for x in inputs]))[i]))

def classify_record(root, record):
    if root.label is not None:
        return root.label
    index = root.attribute_index
    #print(index)
    #print(root.attribute_value)
    branches = root.branches
    for branch in branches:
        if branch.attribute_value == record.data[index]:
            #print(branch.attribute_value)
            return classify_record(root=branch, record=record)

def standard_data(num_of_bins):
    data = read_file()
    normalize_data(data)
    
    for j in range(len(data[0].data)):
        if type(data[0].data[j]) is not str:
            min_elem, max_elem = None, None

            for i in range(len(data)):
                if min_elem is None:
                    min_elem = min(list(zip(*[x.data for x in data]))[j])
                    max_elem = max(list(zip(*[x.data for x in data]))[j])
                bin_width = (max_elem - min_elem) / num_of_bins

                for p in range(1, num_of_bins+1):
                    if data[i].data[j] <= p * bin_width:
                        data[i].data[j] = p
                        break

    get_columns(data)
    return data

def build_tree(input_data, attributes_list):
    #print('truth values', [x.truth for x in input_data])
    classes = set([x.truth for x in input_data])
    #print(classes)
    root = TreeNode()
    #print(len(classes))
    #print(attributes_list)
        
    if len(classes) == 1:
        root.label = classes.pop()
        #print('len(classes)=1',root.label)
    elif len(attributes_list) == 0:
        root.label = get_majority_label(input_data)
    else:
        attribute_index = best_index(input_data, attributes_list)
        #print(attribute_index)
        root.attribute_index = attribute_index
        
        #print(columns[attribute_index].choices)

        for choice in columns[attribute_index].choices:
            branch = TreeNode()
            #print(choice)
            #print(branch.attribute_value)
            choice_subset = [x for x in input_data if x.data[attribute_index] == choice] #entries of choices in the entire input_data
            
            if len(choice_subset) == 0:
                branch.label = get_majority_label(input_data)
                branch.attribute_value = choice
            else:
                #print(len([x for x in attributes_list if x != attribute_index]))
                branch = build_tree(choice_subset, [x for x in attributes_list if x != attribute_index])
                branch.attribute_value = choice
                #print(branch.attribute_index, branch.attribute_value)
            root.add_child(branch)
    #print(root.attribute_index)
    return root

def split_len(a, n):
    k, m = divmod(len(a), n)
    return [len(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]) for i in range(n)]

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def classify_forest(forest, input_data):
    tp, fn, fp, tn = 0, 0, 0, 0
    predictions = []
    for record in input_data:
        record_predictions = []
        for tree in forest:
            root = copy.deepcopy(tree)
            node = copy.deepcopy(root)
            record_predictions.append(classify_record(root=node, record=record))
        predictions.append(statistics.mode(record_predictions))

    for prediction, record in zip(predictions, input_data):
        if prediction == record.truth:
            if prediction == 1:
                tp += 1
            else:
                tn += 1
        else:
            if prediction == 1:
                fp += 1
            else:
                fn += 1
    return tp, fn, fp, tn

def run_random_forests(num_of_bins, num_of_trees, num_of_folds):
    data = standard_data(num_of_bins)
    all_splits = split(data, num_of_folds)

    tp = fn = fp = tn = 0
    
    for i in range(num_of_folds):
        testing_data, training_data = all_splits[i], []
        for sub_list in all_splits:
            if sub_list != testing_data:
                training_data += sub_list

        split_lens = split_len(training_data, num_of_trees)

        forest_trees = []
        for j in range(num_of_trees):

            forest_training_data = []
            len_td = len(training_data)
            while split_lens[j] != 0:
                forest_training_data.append(training_data[randint(0, len_td - 1)])
                split_lens[j] -= 1

            total_attributes = len(columns)

            num_of_attributes_to_be_selected = math.ceil(total_attributes * 0.2)

            selected_attributes = []
            while num_of_attributes_to_be_selected != 0:
                p = randint(0, total_attributes-1)
                if p not in selected_attributes:
                    selected_attributes.append(p)
                    num_of_attributes_to_be_selected -= 1

            forest_trees.append(build_tree(forest_training_data, selected_attributes))
        
        a, b, c, d = classify_forest(forest_trees, testing_data)

        tp += a
        fn += b
        fp += c
        tn += d

        accuracy = (tp/num_of_folds + tn/num_of_folds) / (tp/num_of_folds + fn/num_of_folds + fp/num_of_folds + tn/num_of_folds)
        precision = (tp/num_of_folds) / (tp/num_of_folds + fp/num_of_folds)
        recall = (tp/num_of_folds) / (tp/num_of_folds + fn/num_of_folds)
        f_measure = (2 * tp/num_of_folds) / (2 * tp/num_of_folds + fn/num_of_folds + fp/num_of_folds)
        
        print('--------- Performance Measures after', (i+1), 'Folds ----------------')
        print('Accuracy:', accuracy)
        print('Precision:', precision)
        print('Recall:', recall)
        print('F-Measure:', f_measure)
        print('')
    
    print('-------------- Final Performance Measures -----------------')
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F-Measure:', f_measure)

run_random_forests(num_of_bins=5, num_of_trees=15, num_of_folds=10)