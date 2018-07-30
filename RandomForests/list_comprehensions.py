# example 1: Flatten a matrix

# using a for loop
def eg1_for(matrix):
    flat = []
    for row in matrix:
        for x in row:
            flat.append(x)
    return flat

# using list comprehension - this is doing the same thing as above
# reads from right --> left: So for each element in the a row in a matrix
# append this to an empty list and the look at each row in matrix

def eg1_lc(matrix):
    return [x for row in matrix for x in row]

# you can define your parameters outside the function

matrix = [range(0,5), range(5,10), range(10,15)]

# print statements for the various outputs or results

print("Original Matrix:"  + str(matrix))
print("FOR LOOP Result:" + str(eg1_for(matrix)))
print("LC Result:" + str(eg1_lc(matrix)))

# Example 2: Removing vowels from a sentence
# Aim: Take a string as input and return a string with vowels removed.

def eg2_for(sentence):
    vowels = 'aeiou'
    filtered_list =[]
    for l in sentence:
        if l not in vowels:
            filtered_list.append(l)
    return ''.join(filtered_list)

def eg2_lc(sentence):
     vowels = 'aeiou'
     return ''.join([l for l in sentence if l not in vowels])

sentence = 'My name is Aarshay Jain!'
print("FOR-loop result: " + eg2_for(sentence))
print("LC result      : " + eg2_lc(sentence))

# Example 3: Dictionary Comprehension
# Aim: Take two list of same length as input and return a dictionary 
# with one as keys and other as values.
# Python codes with FOR-loop and LC implementations:

def eg3_for(keys, values):
    dic ={}
    for i in range(len(keys)):
        dic[keys[i]] = values[i]
    return dic

def eg3_LC(keys,values):
    return{keys[i]:values[i] for i in range(len(keys))}

# lets define the lists with keys and values

country = ['India', 'Pakistan', 'Nepal', 'Bhutan', 'China', 'Bangladesh']
capital = ['New Delhi', 'Islamabad','Kathmandu', 'Thimphu', 'Beijing', 'Dhaka']

print("FOR-loop result: " + str(eg3_for(country, capital)))
print("LC result      : " + str(eg3_LC(country, capital)))


