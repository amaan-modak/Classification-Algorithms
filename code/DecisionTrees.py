import copy
import math
import statistics

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
    input_file = 'project3_dataset4.txt' #Change this for changing input file
    array = []

    with open(input_file) as file:
        for row in file:
            data = row.split('\t')
            data = [float(x) if is_number(x) else x for x in data]
            data_row = DataRow(data[-1], data[:-1])     
            array.append(data_row)
    return array

def split_data(data, split_value):
    training_set = math.floor(split_value * len(data))
    return data[:training_set], data[training_set:]

def normalize_data(inputs):
    for j in range(len(inputs[0].data)):
        if type(inputs[0].data[j]) is not str:
            min_elem, max_elem = None, None
            for i in range(len(inputs)):
                if min_elem is None:
                    min_elem = min(list(zip(*[x.data for x in inputs]))[j])
                    max_elem = max(list(zip(*[x.data for x in inputs]))[j])
                inputs[i].data[j] = (inputs[i].data[j] - min_elem) / (max_elem - min_elem)

def entropy(inputs):
    positives = inputs.count(1.0)
    negatives = len(inputs) - positives
    total = len(inputs)

    if positives == 0 or negatives == 0:
        return 0
    else:
        entropy = -(positives/total) * math.log(positives/total,2) - (negatives/total) * math.log(negatives/total,2)
        #print(entropy)
        return entropy

def information_gain(input_data, attribute_index):
    subset_entropy = 0
    for choice in columns[attribute_index].choices:
        input_data_subset = [x.truth for x in input_data if x.data[attribute_index] == choice]
        subset_entropy += (len(input_data_subset) / len(input_data)) * entropy(input_data_subset)
    input_data = [x.truth for x in input_data]
    res_entropy = entropy(input_data) - subset_entropy
    #print(res_entropy)
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
        #print(root.label)
        return root.label
    index = root.attribute_index
    branches = root.branches
    for branch in branches:
        if branch.attribute_value == record.data[index]:
            #print(branch.attribute_value)
            return classify_record(root=branch, record=record)

def validation(root, input_data):
    tp, fn, fp, tn = 0, 0, 0, 0
    predictions = []
    for record in input_data:
        node = copy.deepcopy(root)
        predictions.append(classify_record(root=node, record=record))

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
    
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 2 * tp / (2 * tp + fn + fp)

    print('---------------Decision Tree Measures----------------')
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F-Measure:', f_measure)

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
    classes = set([x.truth for x in input_data])
    #print(classes)
    root = TreeNode()

    if len(classes) == 1:
        root.label = classes.pop()
    elif len(attributes_list) == 0:
        root.label = get_majority_label(input_data)
    else:
        attribute_index = best_index(input_data, attributes_list)
        root.attribute_index = attribute_index
        #print(root.attribute_index)
        #print(columns[attribute_index].choices)
        
        for choice in columns[attribute_index].choices:
            #print(choice)
            branch = TreeNode()
            choice_subset = [x for x in input_data if x.data[attribute_index] == choice]

            if len(choice_subset) == 0:
                branch.label = get_majority_label(input_data)
                branch.attribute_value = choice
            else:
                branch = build_tree(choice_subset, [x for x in attributes_list if x != attribute_index])
                branch.attribute_value = choice
            root.add_child(branch)
    return root

def visualize_treenode(root):
    branches = root.branches
    print("Choice : ", root.attribute_value)
    if (len(branches) == 0):
        print("LEAF NODE")
        print("Label : ", root.label)
    else:
        #node_idx = node_idx + 1
        print("Attribute idx: ", root.attribute_index)
        print("Number of branches: ",len(branches))
        for b in range(len(branches)):
            print('')
            print("Branch: ", b+1)
            branchNode = branches[b]
            visualize_treenode(branchNode)

def run_algorithm(split_value, num_of_bins):
    data = standard_data(num_of_bins)
    training_data, testing_data = split_data(data, split_value)
    root = build_tree(training_data, range(len(columns)))
    print('--------- Printing Decision Tree -----------------')
    visualize_treenode(root)
    validation(root, testing_data)

run_algorithm(split_value=0.90, num_of_bins=10) #Split value is the ratio to divide data into training and testing data for validation