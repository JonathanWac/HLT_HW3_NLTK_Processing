#####################################################
# Jonathan Wachholz
# jhw190002
# HLT HW 3
#
# This homework demonstrates more complex Text Preprocessing using the
#  NLTK library. By default it will process the "moby_dict.txt" file located at
#  the same file directory as the python file.
#
#####################################################

import os
import sys
import re
import nltk
from typing import List, Tuple, Dict

stops = set(nltk.corpus.stopwords.words("english"))


def openFile(filepath: str):
    """Function to open a file path that works cross platform"""
    return open(os.path.join(os.getcwd(), filepath), 'r')


def getRawText(filepath: str, skipLines: int = 0) -> str:
    """"
    Function to open a text file and return the raw text from that file...
    :param filepath: The path location of the file
    :param skipLines: The number of lines of the text file to skip over when reading text from the file

    :returns The raw text of the file
    """
    text = str()
    with openFile(filepath) as inFile:
        for i in range(skipLines):
            inFile.readline()
        text += inFile.read()
    return text


def processText(text: str):
    """"
        Function to process raw text to make it all lowercase, remove "--", all digits, and all punctuation
        :param text: The text to be processed

        :returns The processed text
    """
    # 2a
    text = text.lower()
    # 2b
    text = text.replace("--", "")
    # 2c
    text = re.subn(r"\d+", "", text)
    numChanges = text[1]
    text = text[0]
    print(f"\t{numChanges} sets of digits removed...")
    # 2d
    punctuationPattern = r"[\.\?\!\,\;\:\[\]\{\}\(\)\'\"\-\—\_]+"
    text = re.subn(punctuationPattern, " ", text)
    numChanges = text[1]
    text = text[0]
    print(f"\t{numChanges} sets of punctuation removed...")
    return text


def tokenizeText(text: str) -> list:
    """"
        Function to tokenize an input text using the NLTK libraries word_tokenize function...
        :param text: The text to be processed

        :returns The list of tokens
    """
    tokens = nltk.tokenize.word_tokenize(text)
    print(f"Tokens length: {len(tokens)}")
    return tokens


def uniqueWords(tokens: list) -> set:
    """"
        Function to return a unique set of tokens by converting a list into a set (thus removing all duplicates)
        :param tokens: The tokens to be processed

        :returns The unique set of tokens
    """
    uniqueTokens = set(tokens)
    print(f"Unique Tokens length: {len(uniqueTokens)}")
    return uniqueTokens


def removeStopWords(tokens: list) -> list:
    """"
        Function to remove stop words from a list of tokens (stop words are from nltk.corpus.stopwords.words("english"))
        :param tokens: The list of tokens to be processed

        :returns The list of tokens with stop words removed
    """
    newTokens = [word for word in tokens if word not in stops]
    print(f"Important Tokens length: {len(newTokens)}")
    return newTokens


def stemWords(words: list, stemmerType="porter", DEBUG=False):
    """"
        Function to stem words from a list of tokens using a given stemmer type
        :param words: The list of tokens to be stemmed
        :param stemmerType: A string representing the stemmer type to use
        :param DEBUG: Boolean to turn on debug mode which will print out the word stem list after creation

        :returns A list of tuples, with each tuple composed of (word, stem of word)
    """
    stemmerTypeList = ["lancaster", "snowball", "porter", "wordnet"]
    if stemmerType.lower() not in stemmerTypeList:
        print(f"The stemmer type given: \'{stemmerType}\' was not found in the list... \n"
              f"\tNow using the default stemmer type \'{stemmerTypeList[0]}\'")
        stemmerType = stemmerTypeList[0]

    print(f"Now generating (word, stem of word) list using the {stemmerType} stemmer...")

    if stemmerType == "wordnet":
        wordStems = [(word, nltk.stem.WordNetLemmatizer().lemmatize(word)) for word in words]

    elif stemmerType == "lancaster":
        wordStems = [(word, nltk.stem.lancaster.LancasterStemmer().stem(word)) for word in words]

    elif stemmerType == "snowball":
        wordStems = [(word, nltk.stem.snowball.SnowballStemmer("english").stem(word)) for word in words]

    elif stemmerType == "porter":
        wordStems = [(word, nltk.stem.porter.PorterStemmer().stem(word)) for word in words]
    else:
        wordStems = []
    if DEBUG:
        print(wordStems)

    return wordStems


def genStemDict(tupleList: List[tuple]):
    """"
        Function to generate a dictionary from a list of tuples in the form (word, stem of word)
            The stem of the word will be the key, and the value will be a list of all words with that stem

        :param tupleList: The list of (word, stem of word) pairs to be processed

        :returns The dictionary in the form of: {stem: [word1, word2,...], ...}
    """
    wordStemsDict = dict()
    for word, stem in tupleList:
        if stem not in wordStemsDict.keys():
            wordStemsDict[stem] = [word]
        else:
            wordStemsDict[stem].append(word)
    return wordStemsDict


def calcLevenshteinDistance(wordA: str, wordB: str):
    """"
        Function to calculate the Levenshtein Distance between two words recursively
        :param wordA:
        :param wordB: The two words to calculate the distance between

        :returns the Levenshtein Distance between the 2 words
    """
    lenA = len(wordA)
    lenB = len(wordB)

    if lenA == 0 and lenB == 0:
        return 0
    elif lenA == 0:
        return lenB
    elif lenB == 0:
        return lenA

    elif wordA[0] == wordB[0]:
        return calcLevenshteinDistance(wordA[1:], wordB[1:])

    else:
        return 1 + min(calcLevenshteinDistance(wordA[1:], wordB),
                       calcLevenshteinDistance(wordA, wordB[1:]),
                       calcLevenshteinDistance(wordA[1:], wordB[1:]))


def generateLevenshteinDistDict(inList: List[Tuple[str, List[str]]]):
    """"
        Function to generate a dictionary pairing a stem as the key with a list of tuples
            containing the distance between the key/stem and the words connected to that stem...
        :param inList: The input list of tuples in the form of: (stem, list of words with that stem)

        :returns a dictionary pairing a stem as the key with a list of tuples
            containing the distance between the key/stem and the words connected to that stem...
    """
    levenDict = dict()
    for key, lst in inList:
        if key not in levenDict:
            newList = list()
            for word in lst:
                dist = calcLevenshteinDistance(key, word)
                newList.append(tuple([word, dist]))
            levenDict[key] = newList
        # This else block should never be executed, but it is here just incase
        else:
            for word in lst:
                dist = calcLevenshteinDistance(key, word)
                levenDict[key].append(tuple(word, dist))
    return levenDict


def customCheckLevenDict(inDict: Dict[str, List[str]], wordToCompare: str, keyToCompare: str):
    """"
        Function to compare the Levenshtein Distance between a passed word the other words from a list from inDict

        :param inDict: the Input dictionary in the form of {word stem: [word1, word2, ...], ...}
        :param wordToCompare: The word which will be compared with every other word from the inDict[keyToCompare] list
        :param keyToCompare: The key to the inDict which will get the list of words to compare wordToCompare to.

        :returns a custom list of tuples in the form of (comparedWord, Levenshtein Distance)
    """
    inDict = dict(inDict)
    customList = list()
    customList.append(tuple([wordToCompare, 0]))

    if keyToCompare in inDict.keys():
        for val in inDict[keyToCompare]:
            if val == wordToCompare:
                continue
            dist = calcLevenshteinDistance(wordToCompare, val)
            customList.append(tuple([val, dist]))
    return customList


def customCheckLevenDict2(inDict: Dict[str, List[Tuple[str, int]]], wordToCompare: str, keyToCompare: str):
    """"
        Function to compare the Levenshtein Distance between a passed word the other words from a list from inDict

        :param inDict: the Input dictionary in the form of {word stem: [tuple(word1, int), ...], ...}
        :param wordToCompare: The word which will be compared with every other word from the inDict[keyToCompare] list
        :param keyToCompare: The key to the inDict which will get the list of words to compare wordToCompare to.

        :returns a custom list of tuples in the form of (comparedWord, Levenshtein Distance)
    """

    inDict = dict(inDict)
    customList = list()
    customList.append(tuple([wordToCompare, 0]))

    if keyToCompare in inDict.keys():
        for tple in inDict[keyToCompare]:
            if tple[0] == wordToCompare:
                continue
            dist = calcLevenshteinDistance(wordToCompare, tple[0])
            customList.append(tuple([tple[0], dist]))
    return customList


def countPOSTags(tokensPOSTagged: List[Tuple[str, str]], DEBUG=False):
    """"
        Function to count the amount of POS tags from a given list of word + POS Tag pairs...

        :param tokensPOSTagged: a list of tuples in the form of (word, POS_Tag)
        :param DEBUG: A boolean to turn on debugging mode and print the dictionary after its generation

        :returns the Dictionary using the POS tag as the key and its occurrence count as the value
    """
    posDict = dict()
    fullTags = ['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN', 'DT', 'PRP', ':', 'WP$',
                'NNPS', 'PRP$', 'WDT', '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP',
                'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS']

    for tags in fullTags:
        posDict[tags] = 0
    for tple in tokensPOSTagged:
        if tple[1] not in posDict.keys():
            posDict[tple[1]] = 1
        else:
            posDict[tple[1]] += 1

    if DEBUG:
        print(posDict)
    return posDict


if __name__ == '__main__':
    if len(sys.argv) < 2:
        filePath = 'moby_dick.txt'
        print(f"No argument entered for the file path name... \n\tNow using the default file path: {filePath}\n")
    else:
        filePath = sys.argv[1]
        print(f"\tPath given from system arg is: {filePath}\n")
    # 1
    rawText = getRawText(filePath)

    # 2
    processedText = processText(rawText)

    # 3
    tokenizedText = tokenizeText(processedText)

    # 4
    uniqueTokens = uniqueWords(tokenizedText)

    # 5
    importantWords = removeStopWords(uniqueTokens)

    # 6
    wordStemsTuples = stemWords(importantWords)

    # 7
    wordStemsDict = genStemDict(wordStemsTuples)

    # 8
    print(f"Dict keys: {len(wordStemsDict.keys())}, Dict Vals: {len(wordStemsDict.values())}")

    # 9

    sortedDictList = [(k, wordStemsDict[k]) for k in
                      sorted(wordStemsDict, key=lambda k: len(wordStemsDict[k]), reverse=True)]
    for k, v in sortedDictList[:25]:
        print(f"\t{k}: {v}")

    # 10
    levenDistDict = generateLevenshteinDistDict(sortedDictList)

    compareWord = "continue"
    continueDistances1 = customCheckLevenDict(wordStemsDict, compareWord, "continu")
    print(f"Levenshtein Distances from \'{compareWord}\' to: \n\t{continueDistances1}")

    # continueDistances2 = customCheckLevenDict2(levenDistDict, compareWord, "continu")
    # print(f"Levenshtein Distances from \'{compareWord}\' to: {continueDistances2}")
    # 11
    tagsList = nltk.pos_tag(tokenizedText)

    # 12
    posCounts = countPOSTags(tagsList)
    print(f"POS COUNTS DICT:\n\t{posCounts}")
