import numpy as np
import pandas as pd
import nltk
from enchant.checker import SpellChecker
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')
from nltk.corpus import stopwords
from ast import literal_eval
import re
import sys
import warnings

'''
Message Preprocessing Process

1. filtering message
	(i) Remove git-svn-id: 
		e.g. git-svn-id: https://svn.apache.org/repos/asf/commons/proper/collections/trunk@1492866 13f79535-47bb-0310-9956-ffa450edef68
	(ii) Remove Issue Number : e.g. [CALCITE-968]  [COMPRESS-392]
	(iii) Remove Created by MOE - 
		eg. Created by MOE: https://github.com/google/moe, MOE_MIGRATED_REVID=160142836
	(iv) Remove Developer name
		e.g. 344 [] User-defined aggregate functions with more than one parameter (hzyuemeng1)
	(v) Remove http and email address
	(vi) Other filter
		- Authors
		- Reviewers
		- [...]
		- digitial
		- special character

2.  preprocessing message
	- to lower case
	- remove all '\n'
	- remove all bad symbol - special characters
	- remove all stopwords in english

3. Spell Checking - remove project name/class name/method name
	Three Options:
	- keep all wrong words
 	- remove all wrong words
	- replace wrong words with some new words

4. Stemming - Snowball Stemming using NLTK

'''

# filter global
# message filter
GIT_SVN_RE = re.compile('git-svn-id:.*\n')
issue_number_re = re.compile('[A-Za-z]+[-]\d+')
created_by_moe = re.compile('Created by MOE:.*\n')
MOE_ID = re.compile('MOE_MIGRATED_REVID=.*\n')
developer_name_re = re.compile('\(.*\)\n')
HTTP_RE = re.compile('(http|https):.*\n')
EMAIL_RE = re.compile('[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z0-9-.]+')
Other_RE = re.compile('(Author|Authors|Reviewers|Reviewer).*\n')
Bracket_RE = re.compile('\[.*\]')

special_character_re = re.compile(r'[?|!|\'|"|#]')
punctuation = re.compile(r'[.|,|)|(|\|/]')
break_line_re = re.compile('\n')
digital = re.compile('[0-9]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z ]')
STOPWORDS = set(stopwords.words('english'))
# filtering text
def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = GIT_SVN_RE.sub('',text)
    text = issue_number_re.sub('',text)
    text = created_by_moe.sub('',text)
    text = MOE_ID.sub('',text)
    text = developer_name_re.sub('\n', text)
    text = HTTP_RE.sub('\n', text)
    text = EMAIL_RE.sub('\n',text)
    text = Other_RE.sub('', text)
    text = Bracket_RE.sub('', text)

    text = text.lower() # lowercase text
    text = special_character_re.sub('',text)
    text = punctuation.sub(' ',text)
    text = text.strip()
    text = break_line_re.sub(' ',text) # remove all \n
    text = digital.sub('',text)
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text

    text = ' '.join([x for x in text.split() if x and x not in STOPWORDS]) # delete stopwords from text
    return text

# function of wrong spelling words
def check_wrong_words(commit_message):
    chkr = SpellChecker("en_US")
    wrong_words = []
    for index, text in enumerate(commit_message):
        chkr.set_text(text)
        for err in chkr:
            if err.word not in wrong_words:
                wrong_words.append(err.word)
            #   print(index, err.word)
    return wrong_words


def remove_noise_words(message_array, wrong_words):
    res = []
    for text in message_array:
        text = ' '.join([x for x in text.split() if x and x not in wrong_words])
        res.append(text)
    return res

def replace_nosie_words_with_sth(message_array, wrong_words):
    res = []
    for text in message_array:
        # replace wrong word with something
        new_text = []
        for x in text.split():
            if x and x in wrong_words:
                new_text.append('something')
            else:
                new_text.append(x)

# stemming
def stemming(message_array):

    stemmer = SnowballStemmer("english") # Choose a language
    # stemmer.stem("countries")
    for index, text in enumerate(message_array):
    #     print(index, df['cmt_msg'][index])
        message_array[index] = ' '.join([stemmer.stem(x) for x in text.split()])
    #     print(index, df['cmt_msg'][index])

    return message_array


# preprocessing message
def message_processing(message_array, remove_noise = True):
    '''
    prepossing message
    
    Args:
        message_array - List of commit message
        remove_noise - Boolean - whether to remove noise words or not
    
    Returns:
        new message array - List of commit message after prepocessing
    '''
    
    #################################################################################
    # text preprocessing process
    #################################################################################
    
    # filtering
    message_array = [text_prepare(line) for line in message_array]
    
    # remove noise word
    wrong_words = check_wrong_words(message_array)
    remove_noise_message = remove_noise_words(message_array, wrong_words)
    original_commit_message = np.copy(message_array)
    
    if remove_noise == True:
        message_array = remove_noise_message
    else:
        message_array = original_commit_message
    
    #stemming
    message_array = stemming(message_array)
    
    return message_array