import os, re, math, csv

features = ['acclaim', 'admire', 'amazing', 'awesome', 'beautiful', 'beauty', 'best', 'better', 'brilliant', 'charisma', 
'complex', 'enjoy', 'enjoyable', 'enjoyed', 'entertaining', 'excellent', 'fabulous', 'fantastic', 'genuinely', 'glorious', 
'inspiring', 'interesting', 'intelligent', 'good', 'great', 'like' 'love', 'memorable', 'nice', 'outstanding', 
'perfectly', 'pleasant', 'pleasure', 'positive', 'quality', 'recommend', 'remarkable', 'rewarding', 'smart', 'special', 
'star', 'strong', 'superbly', 'thumbs', 'touching', 'triumph', 'unbelievable', 'unique', 'very', 'well', 
'absurd', 'abysmal', 'annoying', 'appalling', 'atrocious', 'average', 'awful', 'bad', 'boring', 'cliche', 
'convoluted', 'crap', 'crappy', 'disappointing', 'disaster', 'disgusting', 'down', 'dumb', 'dumbest', 'excruciating', 
'failed', 'fails', 'flawed', 'flaws', 'forcing', 'horrible', 'incredulous', 'irrelevant', 'lack', 'lacklustre', 
'low', 'mess', 'not', 'overused', 'painful', 'poor', 'poorly', 'predictable', 'shit', 'stereotypical', 
'stupid', 'unfortunately', 'waste', 'wasteful', 'weak', 'unbelievable', 'unconvincing', 'unimaginative', 'worthless', 'worst']


def text_cleaner(string):
    """
    Returns a list of words, defining words as separated on white space.
    Strips out empty strs from the list, makes lowercase and removes punctuation.
    """
    string = re.sub(r'[^\w\s]',"",string)
    string = string.lower().replace('\n',' ').split(" ")
    return [x for x in string if x!='']

def extract_train_data():
    ''' Extract and format the train data into a single list. '''
    data = []
    for file in os.listdir('data/train/pos'):
        path = 'data/train/pos/'+file
        f = open(path, errors='ignore')
        contents = text_cleaner(f.read())
        f.close()
        info = {"text": contents, "pos": 1}
        data.append(info)
    for file in os.listdir('data/train/neg'):
        path = 'data/train/neg/'+file
        f = open(path, errors='ignore')
        contents = text_cleaner(f.read())
        f.close()
        info = {"text": contents, "pos": 0}
        data.append(info)
    return data

def extract_test_data():
    ''' Extract and format the test data into a list.'''
    data = []
    for file in os.listdir('data/test'):
        path = 'data/test/'+file
        f = open(path, errors='ignore')
        contents = text_cleaner(f.read())
        f.close()
        info = {"text": contents, "file": file[:-4]}
        data.append(info)
    return data

def get_probabilities(data):
    '''Go through the data and get the feature and output probabilities'''
    y1 = total = 0
    x_y1 = [0]*len(features)
    x_y0 = [0]*len(features)
    for entry in data:
        total = total + 1
        if entry['pos'] == 1:
            y1 = y1 + 1
        i = 0
        for w in features:
            if w in entry['text']:
                if entry['pos'] == 1:
                    x_y1[i] = x_y1[i] + 1
                else:
                    x_y0[i] = x_y0[i] + 1
            i = i + 1
    y0 = total - y1
    p_y1 = y1/total
    p_x_y1 = []
    for i in range(len(x_y1)):
        p_x_y1.append((x_y1[i]+1)/(y1+2))
    p_x_y0 = []
    for i in range(len(x_y0)):
        p_x_y0.append((x_y0[i]+1)/(y0+2))
    return p_y1, p_x_y1, p_x_y0

def predict(p_y1, p_x_y1, p_x_y0, text):
    ''' Predict whether a review is positive or negative. '''
    sum = 0
    x = []
    for w in features:
        if w in text:
            x.append(1)
        else:
            x.append(0)
    for j in range(len(x)):
        sum = sum + (x[j]*math.log(p_x_y1[j]/p_x_y0[j], 10)) + ((1-x[j])*math.log((1-p_x_y1[j])/(1-p_x_y0[j]), 10))
    sum = sum + math.log(p_y1/(1-p_y1))
    if sum >= 0:
        return 1
    else:
        return 0

def train_accuracy(p_y1, p_x_y1, p_x_y0, data):
    '''Run prediction on trin data and compare to actual classification.'''
    correct_predictions = 0
    for i in range(len(data)):
        prediction = predict(p_y1, p_x_y1, p_x_y0, data[i]['text'])
        if data[i]['pos'] == prediction:
            correct_predictions = correct_predictions + 1
    accuracy = correct_predictions/len(data)
    return accuracy

train_data = extract_train_data()
p_y1, p_x_y1, p_x_y0 = get_probabilities(train_data)

#accuracy = train_accuracy(p_y1, p_x_y1, p_x_y0, train_data)
#print(accuracy)

test_data = extract_test_data()

with open('naive_bayes_predictions.csv', 'w', newline="") as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
                            #quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Id', 'Category'])
    for i in range(len(test_data)):
        prediction = predict(p_y1, p_x_y1, p_x_y0, test_data[i]['text'])
        filewriter.writerow([test_data[i]['file'], prediction])
