# 4. Using this list and data comprehensiions in Data Science

# You might have faced situations in which each cell of a column in a
# data set contains a list. I have tried to simulate this situation
# using a simple example based on this data.
# This contains a dataframe having 2 columns:

# personID: a unique identifier for a person
# skills: the different games a person can play

# load data
import pandas as pd

data = pd.read_csv(
    "/Users/josephlefournour/Projects/VSCode/GithubRepo_test_project/test_project/RandomForests/skills.csv")

print(data)

# create a new column for each sport and give it a 1 or 0

#Split text with the separator ';'
data['skills_list'] = data['skills'].apply(lambda x: x.split(';'))
print(data['skills_list'])


# Next, we need a unique list of games to identify the different number of columns required. 
# This can be achieved through set comprehension. 
# Sets are collection of unique elements

# initialise the set - this is done creating a new set variable
skills_unique = set()

#Update each entry into set. Since it takes only unique value, 
# duplicates will be ignored automatically.

skills_unique.update((sport for l in data['skills_list'] for sport in l))

print(skills_unique)

#Note that, here we have used generator expression so that each value gets updated on the fly 
# and storing is not required. Now we will generate a matrix using LC containing 5 columns 
# with 0-1 tags corresponding to each sport.

#convert set to lists
skills_unique = list(skills_unique)

sport_matrix = [[1 if skill in row else 0 for skill in skills_unique] for row in data['skills_list']]

print(sport_matrix)

# The last step is to convert this into Pandas dataframe’s column 
# which can be achieved as:

data = pd.concat([data, pd.DataFrame(sport_matrix, columns = skills_unique)], axis = 1)
print(data)
