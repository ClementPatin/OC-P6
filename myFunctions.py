def bestDtype(series):
    """
    returns the most memory efficient dtype for a given Series

    parameters :
    ------------
    series : series from a dataframe

    returns :
    ---------
    bestDtype : dtype
    """
    # imports
    import sys
    import pandas as pd
    import gc

    # create a copy()
    s = series.copy()

    # initiate bestDtype with s dtype
    bestDtype = s.dtype

    # initiate a scalar which will contain the min memory
    bestMemory = sys.getsizeof(s)

    # return "cat" or "datetime" if dtype is of kind 'O'
    if s.dtype.kind == "O":
        # return 'datetime64[ns]' if dates are detected
        if s.str.match(r"\d{4}-\d{2}-\d{2} \d{2}\:\d{2}\:\d{2}").all(axis=0):
            bestDtype = "datetime64[ns]"
        else:
            bestDtype = "category"

    # for numericals
    else:
        # test several downcasts
        for typ in ["unsigned", "signed", "float"]:
            sDC = pd.to_numeric(s, downcast=typ)
            # if downcasted Series is different, continue
            if (s == sDC).all() == False:
                continue
            # get memory
            mem = sys.getsizeof(sDC)
            # if best, update bestDtype and bestMemory
            if mem < bestMemory:
                bestMemory = mem
                bestDtype = sDC.dtype
            del sDC
            gc.collect()

    del s
    gc.collect()
    return bestDtype


######### NLP ###########

import spacy
nlp = spacy.load("en_core_web_sm")






def loadReviews(revPath, busPath, revSize, revStars, random_state = 16) :
    '''
    load a given number of reviews from Yelp json file, each one with a given "stars" score.

    parameters :
    ------------
    revPath - string : path of json file containing reviews
    busPath - string : path of json file containing business infos
    revSive - int : number of reviews desired
    revStars - int or list of int : review score(s) of extracted reviews
    random_state - int or None : for ".sample" pandas method. By default : 16

    return :
    --------
    rev - dataframe : 1 column dataframe with reviews
    '''

    # imports
    import pandas as pd
    import gc

    # first load business dataset to find the business that are restaurants
    # load busPath
    with open(busPath, mode="r", encoding="utf8") as f:
        bus = pd.read_json(
            path_or_buf=f, 
            orient="records", 
            lines=True, 
            nrows=None
        )
    # filter bus on "categories" containing the string "restaurant"
    maskResto = bus["categories"].apply(lambda x : "restaurant" in x.lower() if x else False)
    # store corresponding "business_id"s
    busIdsRestau = bus["business_id"].loc[maskResto].values

    del bus
    gc.collect()

    # the load reviews dataset
    # initiate a dataframe
    rev = pd.Series(name="text")
    
    # create an iterator using "chunksize" option, process it to keep only bad reviews and sample it
    # the chunksize used will be equal to revSize
    with open(revPath, mode="r", encoding="utf8") as f:
        # create the reader object
        reader = pd.read_json(
            path_or_buf=f, 
            orient="records", 
            lines=True, 
            chunksize=revSize,
        )

        # handle revStars
        if type(revStars) == int :
            revStars = [revStars]
            
        # iterate on the reader    
        for i,chunk in enumerate(reader):
            chunk = pd.DataFrame(chunk)
            # filter to keep only bad reviews with "stars" in revStars list
            # filter also on selected "business_id"
            maskStars = chunk["stars"].isin(revStars)
            maskResto = chunk["business_id"].isin(busIdsRestau)
            filtered_chunk = chunk.loc[maskStars & maskResto]
            # keep only the "text" column
            light_chunk = filtered_chunk["text"]
            # sample with 1/20 of the target sample size (so we do 20 iteration to reach it)
            light_chunk_samp = light_chunk.sample(int(revSize/20), random_state=random_state)
            # add to rev dataframe
            rev = pd.concat([rev,light_chunk_samp])
            
            # break when enought reviews
            if len(rev) >=  revSize :
                break
    
            del maskStars, maskResto, filtered_chunk, light_chunk, light_chunk_samp
            gc.collect()

    return rev


### create a class for  loadReviews function (for scikit learn pipeline)
# import packages
from sklearn.base import BaseEstimator, TransformerMixin

class reviewsLoader (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a Yelp reviews loader using the loadReviews function
    '''
    def __init__(self, busPath, revSize, revStars, random_state=16) :
        '''
        create the loader
        '''
        self.busPath = busPath
        self.revSize = revSize
        self.revStars = revStars
        self.random_state = random_state

    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call loadReviews function
        '''
        return loadReviews(revPath=X, busPath=self.busPath, revSize=self.revSize, revStars=self.revStars, random_state=self.random_state)







def tokenize(doc, regex) :
    '''
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    '''

    # imports
    import nltk
    from nltk.tokenize import RegexpTokenizer

    # tokenize
    tokenizer = RegexpTokenizer(regex)
    tokens = tokenizer.tokenize(doc)

    return tokens



def myLower(textSeries) :
    '''
    given a text Series, put each doc to lower case

    parameters :
    ------------
    textSeries - Series : each value is a text document (string type)
    
    '''
    # imports 
    import pandas as pd

    # put each doc to lower case
    return textSeries.apply(lambda doc : doc.lower().strip())

                            


### create a class for myLower function (for scikit learn pipeline)

class lowerCasor (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a text Series transformer which put each value too lowercase using the myLower function
    '''
    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call myLower function
        '''
        return myLower(textSeries=X)







def removeURL(textSeries) :
    '''
    given a text Series, remove url 

    parameters :
    ------------
    textSeries - Series : each value is a text document (string type)

    return :
    --------
    out - Series : the same but without url
    
    '''
    # imports 
    import pandas as pd
    import re

    # remove escape sequences
    out = textSeries.apply(lambda doc : re.sub(r'https?:[^\s]+',' ',doc))
    return out

### create a class for removeURL function (for scikit learn pipeline)

class urlRemover (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a URL remover using the removeURL function
    '''
    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call removeURL function
        '''
        return removeURL(textSeries=X)






def removeEscapeSequences(textSeries) :
    '''
    given a text Series, remove escape sequences 

    parameters :
    ------------
    textSeries - Series : each value is a text document (string type)

    return :
    --------
    out - Series : the same but without escape sequences
    
    '''
    # imports 
    import pandas as pd
    import re

    # remove escape sequences
    out = textSeries.apply(lambda doc : re.sub(r'[\n]|[\r]|[\a]|[\b]|[\\]|[\f]|[\t]|[\v]',' ',doc))
    return out

### create a class for removeEscapeSequences function (for scikit learn pipeline)

class escapeSequencesRemover (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a escape sequences remover using the removeEscapeSequences function
    '''
    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call removeEscapeSequences function
        '''
        return removeEscapeSequences(textSeries=X)








def findMultipleChars(text, n):
    '''
    given string, return a list a all words containing repeated characters

    parameters :
    ------------
    text - string
    n - int : the number of times a character should be repeated to be indeed considered a repeated character
    '''
    # imports
    import re

    # find n-times repeated letters (lowercase or uppercase)
    iterator = re.finditer(r'\b\w*([a-zA-Z])\1{'+str(n-1)+',}\w*', text)
    
    # return the whole string of each
    return [match.group() for match in iterator]



def dropDuplicatedChars(text, n, keepDouble=False):
    '''
    given a string, remove repeated characters

    parameters :
    ------------
    text - string
    n - int : the number of times a character should be repeated to be indeed considered a repeated character
    keepDouble : wether or not to leave to 2 chars in place of 1

    return :
    --------
    out - string : the same string, but without repeated chars
    
    
    '''
    # imports
    import re
    # handle keepDouble
    if keepDouble :
        repl = r'\1\1'
    else :
        repl = r'\1'

    # replace "n-times" repeated chars
    out = re.sub(
        pattern = r'([a-z]|[0-9])\1{'+str(n-1)+',}',
        repl = repl,
        string = text
                )
    
    return out



def correctRepeatedChars_text(text) :
    '''
    given a text, remove repeated characters 

    parameters :
    ------------
    text - string 

    return :
    --------
    out - string : the same but with words with repeated characters corrected
    
    '''

    # imports
    import spacy
    from nltk.corpus import wordnet
    import re

    # find words with 3+ repeated chars using findMultipleChars
    find = findMultipleChars(text,3)

    # if no repeated chars, return text
    if len(find) == 0 :
        return text

    else :
        # initiate a dict with corrections
        corrections = {}

        # find a correction for each word with duplicated chars
        for weirdWord in find :
            # possible solutions to test
            possibleSolutions = [
                weirdWord, # maybe the repeated chars are normal
                dropDuplicatedChars(weirdWord, 2), # remove all double chars
                dropDuplicatedChars(weirdWord, 3), # remove all triple chars
                dropDuplicatedChars(weirdWord, 3, keepDouble=True) # remove all triple chars, but keep 2
            ]

            for word in possibleSolutions :
                # if word is in wordnet, keep it
                if word in wordnet.words() :
                    corrections[weirdWord] = word
                    break
                # if not, check its lemma
                else :
                    # compute lemma
                    lemma = nlp(word)[0].lemma_
                    # if lemma is in wordnet, keep it
                    if lemma in wordnet.words() :
                        corrections[weirdWord] = lemma
                        break

        # replace each word with repeated chars with its correction
        for pb,corr in corrections.items() :
            text = re.sub(
                pattern = pb,
                repl = corr,
                string = text
                )
        return text



def correctRepeatedChars_series (textSeries) :
    '''
    given a text Series, correct words with repeated characters using correctRepeatedChars_text function

    parameters :
    ------------
    textSeries - Series : each value is a text document (string type)

    return :
    --------
    correctedSeries - Series : the same but with words corrected
    
    '''
    # imports 
    import pandas as pd
    
    # use correctRepeatedChars_text function to correct these words
    correctedSeries = textSeries.apply(lambda doc : correctRepeatedChars_text(doc))
    
    return correctedSeries


### create a class for correctRepeatedChars_series function (for scikit learn pipeline)

class repeatedCharsCorrector (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a  repeated characters corrector using the correctRepeatedChars_series function
    '''
    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call correctRepeatedChars_series function
        '''
        return correctRepeatedChars_series(textSeries=X)






def cleanText(textSeries, customStopWords = [], posToKeep = ["NOUN","ADJ"]) :

    '''
    given a text Series, apply cleaning methods using Spacy tools :
                - tokenization
                - remove punctuation
                - remove stopwords
                - remove digits and digit-likes ("ten")
                - normalization (lemmatization)
                - filter on POS (part-of-speech) tagging
                - remove custom stopwords

    parameters :
    ------------
    textSeries - Series : each value is a text document (string type)
    customStopWords - list of strings : list of custom stop-words to remove.By default : [], no removal
    posToKeep - list of strings : list of pos tags to keep (Spacy likes). By default : ["NOUN","ADJ"]

    returns :
    ---------
    outText - Series of strings : the same one, after cleaning
    outTokens - Series of list of strings : dito but tokenized
    '''

    # imports
    import pandas as pd

    # create an output list for modified texts
    outText = []
    # create an output list of final tokens
    outTokens = []
    # pass text through pipeline
    for doc in nlp.pipe(
        texts=textSeries.values,
        disable=['ner']
    ) :
        
        # get rif of stopwords, puntuation, spaces, digits, num-likes and custom stopwords
        tokens = [
            token.lemma_ for token in doc \
            if \
            (
                not \
                token.is_stop
                | token.is_punct
                | token.is_space
                | token.is_digit
                | token.like_num
            )
            and
            (
                token.pos_ in posToKeep
            )
            and
            (
                token.lemma_ not in customStopWords
            )
        ]
        # append this doc to the lists
        outText.append(" ".join(tokens))
        outTokens.append(tokens)
    
    
    # as a Series
    outText = pd.Series(outText, index = textSeries.index)
    outTokens = pd.Series(outTokens, index = textSeries.index)
    
    return outText, outTokens
    
### create a class for  cleanText function (for scikit learn pipeline)
class textCleaner (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, cleaner for Yelp reviews using function cleanText
    '''
    def __init__(self, customStopWords = [], posToKeep = ["NOUN","ADJ"]) :
        '''
        create the cleaner
        '''
        self.customStopWords = customStopWords
        self.posToKeep = posToKeep

    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call cleanText function AND return only tokenized output
        '''
        return cleanText(textSeries=X, customStopWords=self.customStopWords, posToKeep=self.posToKeep)[1]







def makeDictionaryAndFilter(tokensSeries, no_below = 0, no_above = 1.0) :
    '''
    use gensim Dictionary function on a Series of tokens, and apply frequency filtering

    parameters :
    ------------
    tokensSeries - Series of list of strings : each value is a list of text document tokens
    no_below - int : filter out tokens that appear less than `no_below` documents (absolute number). By default : 0  (no filtering)
    no_above - float : more than `no_above` documents (fraction of total corpus size, *not* absolute number). By default : 1.0  (no filtering)

    returns :
    ---------
    dictionary - gensim dictionary object
    
    '''

    # imports
    import pandas as pd
    from gensim.corpora import Dictionary

    # create dictionary
    dictionary = Dictionary(tokensSeries.values)

    # filter
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    return dictionary



def makeBOW(tokensSeries, dictionary) :
    '''
    create a bag-of-words vector from a given Series of tokens and a gensim dictionary

    parameters :
    ------------
    tokensSeries - Series of list of strings : each value is a list of text document tokens
    dictionary - gensim dictionary object : created with tokensSeries

    return :
    --------
    bow_vector - list of bow : each bow is itself a list of (token_id, token_count) 2-tuples
    
    '''
    # imports
    import pandas as pd
    from gensim.corpora import Dictionary

    # create bow vector
    bow_vector = [dictionary.doc2bow(text) for text in tokensSeries.values]

    return bow_vector








def makeTFIDF(tokensSeries, dictionary, smartirs = 'nfc') :
    '''
    create a bag-of-words vector AND apply TFIDF from a given Series of tokens and a gensim dictionary
    
    parameters :
    ------------
    tokensSeries - Series of list of strings : each value is a list of text document tokens
    dictionary - gensim dictionary object : created with tokensSeries
    smartirs - string - XYZ form :
            X : term frequency weighting. By default : n (raw term frequency, i.e. count)
            Y : document frequency weighting. By default : f (inverse collection frequency, with log2)
            Z : document normalisation. By default : c (cosine normalisation, i.e. l2 norm)

    return :
    --------
    tfidf_vector - list of bag-of-weights : each one is itself a list of (token_id, token_weight - float) 2-tuples
    
    '''
    # imports
    from gensim.models import TfidfModel
    import pandas as pd
    from gensim.corpora import Dictionary

    # create bow vector
    bow_vector = [dictionary.doc2bow(tokens) for tokens in tokensSeries.values]
    
    # create and fit the tfidf model
    model = TfidfModel(
        dictionary=dictionary,
        smartirs=smartirs
    ) 
    
    # apply to our corpus
    tfidf_vector = [model[bow] for bow in bow_vector]

    return tfidf_vector




### create a class for  makeDictionaryAndFilter, makeBow and makeTFIDF functions (for scikit learn pipeline)
class dictAndVectorMaker (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, dictionary and bag-of-words vector maker for Yelp reviews using functions makeDictionaryAndFilter and makeBOW or makeTFIDF
    '''
    def __init__(self, no_below = 0, no_above = 1.0, smartirs = 'nfc', applyTfidf = False) :
        '''
        create the maker
        new parameter : applyTfidf - bool : wether or not to apply tfidf weighting 
        '''
        self.no_below = no_below
        self.no_above = no_above
        self.smartirs = smartirs
        self.applyTfidf = applyTfidf

    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility and dictionary creation using makeDictionaryAndFilter function
        '''
        self.dictionary = makeDictionaryAndFilter(
            tokensSeries=X, 
            no_below=self.no_below, 
            no_above=self.no_above
        )
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call makeBOW function
        '''
        if self.applyTfidf :
            vector = makeTFIDF(tokensSeries=X, dictionary=self.dictionary, smartirs=self.smartirs)
        else :
            vector = makeBOW(tokensSeries=X, dictionary=self.dictionary)
        
        return vector, self.dictionary






def wordCloudAndCoherence(model, corpus) :
    '''
    given a gensim LDA model and a corpus, plot a WordCoud for each topic and compute the coherence score

    parameters :
    ------------
    model - gensim LDA model already fitted
    corpus - list of gensim bow-like : each bow-like is a list of tuples

    output :
    --------
    a matplotlib figure with a wordcloud axes for each topic and the coherence score in the title
    
    '''

    # imports
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import numpy as np
    from gensim.models.coherencemodel import CoherenceModel

    # create a figure
    # retreive the number of topics. It wil be the number of axes
    n = model.num_topics
    # compute nrows and ncols for matplotlib subplots, with max 3 wordcloud per row
    nrows = int(np.ceil(n/3))
    ncols = int(np.ceil(n/nrows))
    # create figure
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14,14/ncols*nrows)
        )
    # reshape axs
    axs = axs.ravel()

    # create wordcloud instance
    wc = WordCloud(
        width = 400,
        height = 400,
        background_color="white",
        colormap="cividis"
        )

    # plot wordclouds
    for topicIdx,ax in enumerate(axs) :
        # remove unecessary axes (if n is odd)
        if topicIdx+1 > n :
            ax.remove()
            continue

        # plot the first 40 words
        ax.imshow(wc.fit_words(dict(model.show_topic(topicIdx,40))))
        # remove axis
        ax.axis("off")
        # add axes title
        ax.set_title("topic "+str(topicIdx))
    
    
    # main title
    # compute coherence score
    cm = CoherenceModel(model=model, corpus=corpus, coherence="u_mass")
    coherenceScore = round(cm.get_coherence(),2)
    # add main title
    fig.suptitle("LDA Topic Modeling with "+str(n)+" topics\nCoherence score = "+str(coherenceScore))
    
    plt.show()






def plotDifferentNtopics(nTopicsList, dictionary, corpus) :

    '''
    compute "u_mass" coherence_score for different parameters "num_topics" and plot results

    parameters :
    ------------
    nTopicsList - list of int : list of "num_topics" for gensim LdaModel
    dictionary - gensim dictionary object : created with tokensSeries
    corpus - list of bow/tfidf : each bow or tfidf is itself a list of (token_id, token_count or wieghts) 2-tuples
    

    return :
    --------
    matplotlib figure object - seaborn lineplot with coherence_score by num_topics
    '''
    
    # imports
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from gensim.models.ldamodel import LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    
    
    # create a list to store coherence scores
    coherenceScores = []

    # iterate on nTopicsList
    for nTopics in nTopicsList :
        # create the LDA model 
        model = LdaModel(
            corpus=corpus,
            num_topics=nTopics,
            id2word=dictionary,
            random_state=16
        )
    
        # compute cohenrence score
        cm = CoherenceModel(model=model, corpus=corpus, coherence="u_mass")
        coherenceScore = round(cm.get_coherence(),2)
    
        coherenceScores.append(coherenceScore)
    
    # coherenceScores = np.random.rand(len(nTopics))
    
    # create a figure
    fig,ax = plt.subplots(1,1,figsize=(14,6))
    
    # plot
    sns.lineplot(x=nTopicsList, y = coherenceScores, ax=ax)
    
    # title
    fig.suptitle("coherence scores by `num_topics`")
    
    # labels
    ax.set_xlabel("Number of topics")
    ax.set_ylabel("Coherence score (u_mass)")
    
    # xticks
    ax.set_xticks(nTopicsList)
    
    return fig










######### Computer Vision ##############

def loadYelpJsonPhotosData(jsonPath, photosDirPath) :
    '''
    load Yelp photos dataset "photos.json" and put it in a dataframe

    parameters :
    ------------
    jsonPath - str : path to the "photos.json" file
    photosDirPath - str : path to the directory containing Yelp photos

    return :
    --------
    photoData - dataframe : with "photos.json" columns and a another with the path of each photo
    '''
    
    # imports
    import pandas as pd

    # load photos data .json in a dataframe
    photoData = pd.read_json(
        path_or_buf=jsonPath,
        orient="records", 
        lines=True, 
        )

    # add a "photo_path" column
    photoData["photo_path"] = photosDirPath + photoData["photo_id"] + ".jpg"

    return photoData



### create a class for  loadYelpJsonPhotosData function (for scikit learn pipeline)
# import packages
from sklearn.base import BaseEstimator, TransformerMixin
# define class
class YelpJsonPhotosDataLoader (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, loader for Yelp photos using function loadYelpJsonPhotosData
    '''
    def __init__(self, photosDirPath) :
        '''
        create the loader
        '''
        self.photosDirPath = photosDirPath

    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call loadYelpJsonPhotosData function
        '''
        return loadYelpJsonPhotosData(jsonPath = X, photosDirPath = self.photosDirPath)





def removePath(path):
    '''
    delete path, file or directory

    parameter :
    -----------
    path - string : folder or file path
    '''
    # import
    import os
    import shutil

    # first check if the path exist and then delete
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))





def sampPhotos(photoDf, nToKeepByClass, photosDirPath, random_state = 16) :
    '''
    sample the Yelp photos data dataframe, keeping a given number of photos for each category

    parameters :
    ------------
    photoDf - dataframe : base Yelp photos data, from "photos.json"
    nToKeepByClass - int or None : number of samples to keep in each class. By default : None (no sampling)
    photosDirPath - string : path to the directory containing Yelp photos
    randam_state - int : radom_state parameter of the .sample method. By default : 16

    return :
    --------
    photoDfSamp - dataframe : same dataframe, but sampled
    '''
    # imports
    import pandas as pd
    import os
    import gc

    # check available photos in directory and, if necessary, update photoDf
    # first put in a Series the photos names (without ".jpg")
    availablePhotosIds = pd.Series(os.listdir(photosDirPath)).str.split(".jpg").apply(lambda x : x[0])
    # the filter photoDf on this ids
    mask = photoDf["photo_id"].isin(availablePhotosIds.values)
    photoDfUpdated = photoDf.loc[mask]

    # agregate on "label" and take nToKeepByClass samples :
    if nToKeepByClass :
        photoDfSamp = photoDfUpdated.groupby("label").sample(nToKeepByClass, random_state=random_state)
    else :
        photoDfSamp = photoDfUpdated.copy()

    del photoDfUpdated
    gc.collect()

    return photoDfSamp

### create a class for  sampPhotos function (for scikit learn pipeline)
class photoSampler (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a sampler for Yelp photos using function sampPhotos
    '''
    def __init__(self, nToKeepByClass, photosDirPath, random_state = 16) :
        '''
        create the sampler
        '''
        self.nToKeepByClass = nToKeepByClass
        self.photosDirPath = photosDirPath
        self.random_state = random_state

    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call sampPhotos function
        '''
        return sampPhotos(photoDf = X, nToKeepByClass = self.nToKeepByClass, photosDirPath = self.photosDirPath, random_state = self.random_state)



def deletePhotos(photoDfSamp, photosDirPath, deleteMostPhotos) :
    '''
    delete images to save disk space, if needed

    parameters :
    ------------
    photoDfSamp - dataframe : sampled base Yelp dataframe
    photosDirPath - string : path to the directory containing Yelp photos
    deleteMostPhotos - bool : wether or not to run deleting files

        
    '''
    # imports
    import os

    # create a list of all available photos
    photosToCheck = os.listdir(photosDirPath)
    
    # delete images to save disk space, if needed
    if deleteMostPhotos :
        # delete images not included in the photoDfSamp
        
        # first create a list of photos paths to delete
        # convert to sets
        photosToKeep = set(photoDfSamp["photo_path"].to_list())
        photosToCheck = set(photosToCheck)
        # make difference and cast to list
        photosToDel = photosToCheck - photosToKeep
        photosToDel = list(photosToDel)
        # remove photos
        for path in photosToDel :
            if os.path.isfile(path) :
                removePath(path)
        print("Photos not included in subset have been deleted")
    else :
        print("No photo deleted")

### create a class for  deletePhotos function (for scikit learn pipeline)
class photosDeleter (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a photos deleter using the deletePhotos function
    '''
    def __init__(self, photosDirPath, deleteMostPhotos) :
        '''
        create the sampler
        '''
        self.photosDirPath = photosDirPath
        self.deleteMostPhotos = deleteMostPhotos

    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call sampPhotos function
        '''
        deletePhotos(photoDfSamp = X, photosDirPath = self.photosDirPath, deleteMostPhotos = self.deleteMostPhotos)
        return X


def loadImages(listOfPaths) : 
    '''
    given a list of images paths, load them (RGB) and store in a list

    parameters :
    ------------
    listOfPaths - list of strings : list of images paths
    '''

    # imports
    import cv2
    import numpy as np

    # load images (in BGR, by default with openCV) and store them in a list
    listOfImages = [cv2.imread(path) for path in listOfPaths]

    # convert to RGB
    listOfImages = [cv2.cvtColor(
        src=img,
        code=cv2.COLOR_BGR2RGB
    )
                              for img in listOfImages
                              ]

    return listOfImages

### create a class for  loadImages function (for scikit learn pipeline)
class imagesLoader (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a photos loader using the loadImages function
    '''
    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call loadImages function
        '''
        return loadImages(listOfPaths=X["photo_path"].to_list()) 



def displayImagesFromDict(imgDict, title=None, show=True) :
    '''
    display images contained in a given dictionnary

    parameters :
    ------------
    imgDict - dict : 
                    key : image category
                    value : list of images
    title - str : if given, suptitle. By default : None (no title)
    show - bool : wether or not to use plt.show(). By default : True

    output
    ------
    display images
    
    '''
    
    # imports
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # extract dictionnary caracteristics
    catList = list(imgDict.keys())
    nbCats = len(catList)
    nbImgPerCat = len(list(imgDict.values())[0])
    
    # create a figure
    fig, axs = plt.subplots(
        nrows=nbImgPerCat,
        ncols=nbCats,
        figsize=(
            14,
            14/nbCats*nbImgPerCat
        )
    )
    
    # display 2 images per category
    for i,cat in enumerate(catList) :
        # show images
        for j,img in enumerate(imgDict[cat]) :
            # plot
            # handle grayscale
            if len(img.shape) == 2 :
                axs[j,i].imshow(img, cmap='gray',vmin=0,vmax=255)
            else :
                axs[j,i].imshow(img)
            # set anchor
            axs[j,i].set_anchor("N")
            # remove axis
            axs[j,i].axis(False)
            # title on top axes 
            if j==0 :
                axs[j,i].set_title(cat)
            
    # sup title
    if title :
        fig.suptitle(title)

    if show :
        plt.show()



def myConvertGrayScale(imgList, inputFormat="BGR") :
    '''
    convert a given image to grayscale
    parameter :
    -----------
    imgList - list of images, w x h x 3 channels
    inputFormat - str : can be "RGB" or "BGR". By default : "BGR" (the default output of cv2.imread function)

    return :
    --------
    grayList - list of images, w x h : gray scale images
    '''
    # import
    import cv2

    # handle inputFormat
    if inputFormat == "BGR" :
        code = cv2.COLOR_BGR2GRAY
    elif inputFormat == "RGB" :
        code = cv2.COLOR_RGB2GRAY
    # convert to grayscale
    grayList = [cv2.cvtColor(src=img, code=code) for img in imgList]

    return grayList


### create a class for  myConvertGrayScale function (for scikit learn pipeline)
class grayConverter (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a photos grayscale converter using the myConvertGrayScale function
    '''
    def __init__(self, inputFormat = "BGR") :
        '''
        create the grayscale converter
        '''
        self.inputFormat = inputFormat

    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call myConvertGrayScale function
        '''
        return myConvertGrayScale(imgList = X, inputFormat = self.inputFormat)





def myEqualizeHist(imgList) :
    '''
    histogram egalization of a list of images
    parameter :
    -----------
    imgList - list of images

    return :
    --------
    equalList - list of equalized images
    '''
    # import
    import cv2

    # histogram equilization
    equalList = [cv2.equalizeHist(src=img) for img in imgList]

    return equalList


### create a class for  myEqualizeHist function (for scikit learn pipeline)
class equalizer (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a photos histogram equalizer using the myEqualizeHist function
    '''

    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call myEqualizeHist function
        '''
        return myEqualizeHist(imgList = X)






def myGaussianBlur(imgList, ksize, sigma = 0, borderType = None) :
    '''
    gaussian filter on a given list of images
    parameter :
    -----------
    imgList - list of images
    ksize - tuple : gaussian kernel size
    sigma - float : gaussian kernel standard deviation. By default : 0 (in that case : sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8)
    borderType - openCV BorderType : pixel extrapolation method at borders. By default, None, i.e. : cv2.BORDER_DEFAULT (reflect pixels)

    return :
    --------
    blurList - list of smoothed images
    ''' 
    # import
    import cv2

    # opencv GaussianBlur function
    blurList = [
        cv2.GaussianBlur(src=img, ksize=ksize, sigmaX=sigma, borderType=borderType)
        for img in imgList
        ]

    return blurList


### create a class for  myGaussianBlur function (for scikit learn pipeline)
class blurer (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a photos blurer using the myGaussianBlur function
    '''

    def __init__(self, ksize, sigma = 0, borderType = None) :
        '''
        create the blurer
        '''
        self.ksize = ksize
        self.sigma = sigma
        self.borderType = borderType
    
    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call myGaussianBlur function
        '''
        return myGaussianBlur(imgList = X, ksize = self.ksize, sigma = self.sigma, borderType = self.borderType)




def mySIFT(imgList, nfeatures = 0, nOctaveLayers = 3, contrastThreshold = 0.04) :
    '''
    using opencv functions, perform SIFT on a given list of images and return 
        - the same list of images, but with keypoints drawn
        - a list of the images descriptors

    parameters :
    ------------
    imgList - list of images
    nfeatures - int : The number of best features to retain. The features are ranked by their scores.
        By default : 0
    nOctaveLayers - The number of layers in each octave. 3 is the value used in D. Lowe paper. 
        Nota : The number of octaves is computed automatically from the image resolution
    contrastThreshold - The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. 
        The larger the threshold, the less features are produced by the detector.
    
    '''
    # import
    import cv2
    import gc

    # create a SIFT
    sift = cv2.SIFT_create(nfeatures = nfeatures, nOctaveLayers = nOctaveLayers, contrastThreshold = contrastThreshold)

    # detect keypoints and compute descriptors
    kdList = [sift.detectAndCompute(img, None) for img in imgList]

    # split results
    keyPointsList, descriptorsList = [kd[0] for kd in kdList], [kd[1] for kd in kdList]

    # draw keypoints
    drawnKeyPointsList = [
        cv2.drawKeypoints(
            image=img, 
            keypoints=kp, 
            outImage=img, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
        for img,kp in zip(imgList, keyPointsList)
        ]

    del kdList, keyPointsList
    gc.collect()

    return drawnKeyPointsList, descriptorsList


### create a class for  mySIFT function (for scikit learn pipeline)
class SIFTextractor (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a SIFT features extractor using the mySIFT function
    '''

    def __init__(self, nfeatures = 0, nOctaveLayers = 3, contrastThreshold = 0.04) :
        '''
        create the SIFT feature extractor
        '''
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
    
    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call mySIFT function and return only descriptorsList
        '''
        return mySIFT(imgList = X, nfeatures = self.nfeatures, nOctaveLayers = self.nOctaveLayers, contrastThreshold = self.contrastThreshold)[1]




def makeDictionnary(descriptorsList, n_words="sqrt") :
    '''
    Create a visual words dictionnary using MiniBatchKmeans clustering.
    Each cluster centroid will be a visual word

    parameters :
    ------------
    descriptorsList - list of arrays containing image descriptors
    n_words - int or str : the number of clusters/visual words. By default : "sqrt", use the square root of the total number of descriptors

    return :
    --------
    kmeans - sklearn model : fitted clustering model
    '''

    # imports
    from sklearn.cluster import MiniBatchKMeans
    import numpy as np
    import gc
    
    # put all descriptors together
    allDescriptors = np.concatenate(descriptorsList)

    # handle n_words
    if n_words == "sqrt" :
        n_words = int(np.sqrt(len(allDescriptors)))
    
    # kmeans
    kmeans = MiniBatchKMeans(
        n_clusters=n_words, 
        n_init=15,
        batch_size=14*256,
        random_state=16
    )
    kmeans.fit(allDescriptors)

    del allDescriptors
    gc.collect()

    return kmeans



def makeBOVW(fittedKmeans, descriptorsList) :
    '''
    given a fitted visual dictionary kmeans and the list of images descriptors, create a bag of visual words (BOVW) dataframe

    parameters :
    ------------
    fittedKmeans - sklearn model : fitted clustering model, whose each cluster is a visual word
    descriptorsList - list of arrays containing image descriptors : the same used to create the fittedKmeans

    return :
    --------
    BOVW_df - dataframe : with visual words as columns and one row for each image. Appearance count (histogram) of each visual word in each image
    '''

    # import
    import numpy as np
    import gc
    import pandas as pd

    # extract the number of word in dictionary
    n_words = fittedKmeans.n_clusters

    # for each image, predict the cluster (i.e. the visual word) of each of its descriptors
    # thus, for each image, we got an array of visual word, an "image-document"
    corpus = [fittedKmeans.predict(descriptors) for descriptors in descriptorsList]

    # for each image-document, and for each unique value of visual word, count the frequency
    uniqueAndCountWordsList = [np.unique(image_doc, return_counts=True) for image_doc in corpus]

    # initiate a dataframe to store BOVWs
    BOVW_df = pd.DataFrame(columns=np.arange(n_words))

    # for each image and for each visual word, add count
    for i,(uniques, counts) in enumerate(uniqueAndCountWordsList) :
        BOVW_df.loc[i,uniques] = counts

    # fill NaN with 0
    BOVW_df.fillna(0, inplace=True)

    del n_words, corpus, uniqueAndCountWordsList
    gc.collect()

    return BOVW_df



### create a class for  makeDictionnary and makeBOVW functions (for scikit learn pipeline)
class BOVWmaker (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a Bag-of-Visual-Words maker using the makeDictionnary and makeBOVW functions
    '''

    def __init__(self, n_words="sqrt") :
        '''
        create the BOVW maker
        '''
        self.n_words = n_words
    
    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility, call makeDictionnary function to fit the kmeans used as to create the dictionary
        '''
        self.fittedKmeans = makeDictionnary(descriptorsList = X, n_words = self.n_words)
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call makeBOVW function
        '''
        return makeBOVW(fittedKmeans = self.fittedKmeans, descriptorsList = X)



def gensimVectorsToDf(gensimVectors, gensimDictionary) :
    '''
    in bag of visual words context (i.e with integers as word names), transform a gensim list of dense vectors into a pandas dataframe

    parameters :
    ------------
    gensimVectors - list of dense vectors
    gensimDictionary - gensim dictionary : the one used the create the vectors

    return :
    -------
    gensimDf - dataframe : histograms-like dataframe, with no sparsity
    '''

    # imports
    import pandas as pd
    import numpy as np
    
    # initiate the dataframe with the name (an integer) of each visual word
    gensimDf = pd.DataFrame(columns=range(len(gensimDictionary)))

    # each visual word has been mapped to an integer id. Create a dictionary with correspondances
    inversedMapping = {v:int(k) for k,v in gensimDictionary.token2id.items()}

    # iterate on each vectorized "image-document"
    for i,image_vect in enumerate(gensimVectors) :
        # iterate on each pair of the dense vector
        for pair in image_vect :
            gensimDf.loc[i, inversedMapping[pair[0]] ] = pair[1]
            
    # fill nan with 0
    gensimDf.fillna(0,inplace=True)

    return gensimDf



def myTFIDF(BOVW_df) : 
    '''
    for a given dataframe of bag of visual words, apply TF-IDF weightning
    used formula is TF-IDF = (TF = count in the image) x (IDF = log2(Nimages / imageFrequency) )
    l2 normalization is applied

    parameter :
    -----------
    BOVW_df - dataframe : bag of visual words matrix

    return :
    --------
    tfidf_df - dataframe : same after TF-IDF
    '''
    # imports
    import numpy as np
    import pandas as pd

    # number of images
    N = BOVW_df.shape[0]
    # number of images in which each visual word appears ("DF")
    nImagesContainingEachWord = (BOVW_df > 0).sum(axis=0)
    
    # apply formula
    tfidf_df = BOVW_df.apply(lambda r : r * (np.log2(N / nImagesContainingEachWord)), axis = 1)
    
    # "l2" normalization
    tfidf_df = tfidf_df.apply(lambda r : r/np.linalg.norm(r, ord=2),axis=1)
    
    return tfidf_df


### create a class for  myTFIDF function (for scikit learn pipeline)
class TFIDFweighter (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a weighter using the myTFIDF function
    '''

   
    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call myTFIDF function
        '''
        return myTFIDF(BOVW_df = X)







def myPCA(df, n_components):
    """
    run scaling preprocessing, using sklearn.preprocessing.StandardScaler,
    and PCA, using scikit learn sklearn.decomposition.PCA

    parameters :
    ------------
    df - dataframe : DataFrame on which we want to run the PCA
    q - int or float in [0,1] : number of components of the PCA or percentage of amount of variance

    optionnal parameters :
    ----------------------
    ACPfeatures = list of columns names of df used for PCA. By default None (in that case : all dtype 'float64' columns names)

    outputs :
    ---------
    dfPCA : PCA fitted with scaled df
    df_reduced - dataframe : principal components
    
    """
    # imports
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import pandas as pd
    import gc

    # stores values and index
    X = df.values

    # scale
    scaler = StandardScaler()  # instantiate
    X_scaled = scaler.fit_transform(X)  # fit transform X

    # PCA
    dfPCA = PCA(n_components=n_components)
    dfPCA.fit(X_scaled)

    # reduction on scaled X, as a dataframe
    df_reduced = pd.DataFrame(dfPCA.transform(X_scaled), columns=["C"+str(i) for i in range(dfPCA.n_components_)], index=df.index)

    del scaler, X, X_scaled
    gc.collect()
    
    return dfPCA, df_reduced


### create a class for  myPCA function (for scikit learn pipeline)
class PCAreducer (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a PCA reducer using the myPCA function
    '''

    def __init__(self, n_components) :
        '''
        create the PCA reducer
        '''
        self.n_components = n_components
    
    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call myPCA function and return only the reduction
        '''
        return myPCA(df = X, n_components = self.n_components)[1]





def adjustClusterLabels(clusterSeriesBase, clusterSeriesToAdjust):
    """
    match labels between a base clusters Series and another which needs to be ajusted

    parameters :
    ------------
    clusterSeriesBase - Series : clusters with base labels names
    clusterSeriesToAdjust - Series : clusters with labels names needed to be adjusted

    return :
    --------
    clusterSeriesAjusted - Series : the same, with labels names adjusted
    """

    # imports
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    import numpy as np

    # compute confusion matrix
    confMat = confusion_matrix(y_true=clusterSeriesBase, y_pred=clusterSeriesToAdjust)
    
    # create a correspondance array to reorganize the confusion matrix
    
    # normalize matrix
    confMatNorm = confMat/confMat.sum(axis=0)
    # initiate the correspondance array
    corresp = np.array([-1 for i in range(len(confMat))])
    # find the more significant maximums (using confMatNorm) and put their index (i.e. their true class) in corresp
    maxs = np.sort(confMatNorm.max(axis = 0))[::-1]
    for max in maxs :
        # find the column of this max
        columnIdx = np.where(confMatNorm == max)[1][0]
        # use argmax to find the true class index
        idx = np.argmax(confMat[:,columnIdx])
        # if this class not already in corresp, add it
        if idx not in corresp :
            corresp[columnIdx] = idx
        else :
            continue
    # if there is still "-1" in corresp, replace with the missing label
    if -1 in corresp :
        missingIdx = [i for i in range(len(confMat)) if i not in corresp]
        for idx in missingIdx :
            corresp = np.where(corresp==-1,idx,corresp)

    # use these indexes to switch labels in clusterSeriesToAdjust
    clusterSeriesAjusted = clusterSeriesToAdjust.apply(lambda label : corresp[label])

    
    return clusterSeriesAjusted



def compareCategoriesVSKmeans(imageFeaturesDf, catSeries) :
    '''
    given a dataframe of image features and a true labels, 
        - perform kmeans with n_clusters = number of classes
        - use adjustClusterLabels function so the clusters "match" the categories

    parameters :
    ------------
    imageFeaturesDf - dataframe : images features (or a reduction)
    catSeries - Series : images true categories

    return :
    --------
    resultsDf - dataframe : 1 row per image and 3 columns :
                            "label" : categories
                            "labelCode" : same one, label encoded
                            "cluster" : cluster label from KMeans
    '''

    # imports
    from sklearn.cluster import KMeans
    import pandas as pd
    
    # cast catSeries to category
    catSeries = pd.Series(pd.Categorical(catSeries.values), name="label")
    # number of categories
    n_categories = len(catSeries.cat.categories)

    # kmeans
    kmeans = KMeans(n_init=30, n_clusters=n_categories, random_state=16)
    kmeans.fit(imageFeaturesDf)
    
    # create results dataframe
    resultsDf = catSeries.to_frame()
    resultsDf["labelCode"] = catSeries.cat.codes.astype("category")
    resultsDf["cluster"] = kmeans.predict(imageFeaturesDf)

    # use custom function adjustClusterLabels so the "cluster" matches with "labelCode"
    resultsDf["cluster"] = adjustClusterLabels(clusterSeriesBase=resultsDf["labelCode"],clusterSeriesToAdjust=resultsDf["cluster"])

    return resultsDf


def analyseCategoriesVSClusters(resultsDf) :
    '''
    given a dataframe containing true labels and clusters labels compute ARI and the confusion matrix and display them.

    parameters :
    ------------
    resultsDf - dataframe : 1 row per image and 3 columns :
                            "label" : categories
                            "labelCode" : same one, label encoded
                            "cluster" : cluster label from KMeans

    output :
    --------
    display a seaborn heatmap with the confusion matrix and the ARI
    '''

    # imports
    from sklearn.metrics import confusion_matrix, adjusted_rand_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # compute confusion matrix
    confMat = confusion_matrix(y_true=resultsDf["labelCode"], y_pred=resultsDf["cluster"])

    # compute ARI
    ari = adjusted_rand_score(labels_true=resultsDf["labelCode"], labels_pred=resultsDf["cluster"])

    # plot heatmap
    fig, ax = plt.subplots(1, figsize=(4,4))
    sns.heatmap(
        data=confMat,
        annot=True,
        fmt="0",
        ax=ax,
        cbar=False,
        cmap="plasma"
    )
    # plot ARI as text
    x = resultsDf["labelCode"].nunique()+1
    ax.text(x=x,y=0,s="ARI = "+str(round(ari,2)))
    
    ax.set_yticklabels(labels=resultsDf["label"].cat.categories)
    ax.set_ylabel("Images categories")
    
    ax.set_xlabel("Clusters")

    # title
    fig.suptitle("Images features - clustering with KMEANS\ncategories VS clusters")

    plt.show()




def featureExtractorVGG16(listOfPaths) :
    '''
    given a list of images paths, extract features (dimension 4096) thanks to the penultimate layer of a VGG16

    parameters :
    ------------
    listOfPaths - list of strings : list of images paths


    return :
    --------
    featuresDf - dataframe : (Nimages x 4096) features
    '''

    # imports
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import VGG16
    from keras.models import Model
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions

    import numpy as np
    import pandas as pd
    

    # load images
    photos = []
    
    for path in listOfPaths :
        # load photo
        photo = load_img(path, target_size=(224,224))
        # convert PIL to array
        photo = img_to_array(photo)
        # add to photos list
        photos.append(photo)

    # convert to array
    photos = np.array(photos)

    # preprocess for VGG16 (RGB --> BGR, zero centerage)
    photos = preprocess_input(photos)

    # use pre-trained VGG16 from keras to create a standalone feature extraction model 
    # initiate VGG16
    extractorFromVGG16 = VGG16()
    # remove the output layer (the one with softmax)
    extractorFromVGG16 = Model(inputs=extractorFromVGG16.inputs, outputs=extractorFromVGG16.layers[-2].output)

    # compute features (dim 4096) for each image
    featuresDf = pd.DataFrame(extractorFromVGG16.predict(photos))

    return featuresDf