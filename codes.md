# codes are written in R studio
# identification of spam sms using Naive Bayes classifier
```r
# reading the data and identify the types of sms
sms <- read.csv(file.choose(), stringsAsFactors=F)
str(sms)
round(prop.table(table(sms$type))*100, digits = 1)
sms$type = factor(sms$type) #fix the ‘type’ feature as it’s really a factor with two categories,i.e. spam and ham
```
```r
# building the corpus
library(tm) #install and load tm package
sms_corpus <- Corpus(VectorSource(sms$text)) #build corpus (collection of documents)
print(sms_corpus)
inspect(sms_corpus[1:10])
```
```r
#cleaning the corpus
sms_corpus <- tm_map(sms_corpus, function(x) x <- iconv(x,"WINDOWS-1252","UTF-8"))
corpus_clean <- tm_map(sms_corpus, content_transformer(tolower)) # convert to lower case
corpus_clean <- tm_map(corpus_clean, content_transformer(removeNumbers)) # remove digits
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())# and but etc.
corpus_clean <- tm_map(corpus_clean, removePunctuation) # !
corpus_clean <- tm_map(corpus_clean, stripWhitespace) 
inspect(corpus_clean[1:10]) #reading 1st 10 documents in corpus
```
```r
#breaking each message into words by using document term matrix
dtm<- DocumentTermMatrix(corpus_clean)
str(dtm)
# split the raw data:
sms.train = sms[1:4169, ] # about 75%
sms.test  = sms[4170:5572, ] # the rest

# then split the document-term matrix
dtm.train = dtm[1:4169, ]
dtm.test  = dtm[4170:5572, ]

# and finally the corpus
corpus.train = corpus_clean[1:4169]
corpus.test  = corpus_clean[4170:5572]

#raw data should have more than 80% ham
# in both training and test sets:
round(prop.table(table(sms.train$type))*100)
round(prop.table(table(sms.test$type))*100)
```
```r
#DTMs have more than 7000 columns - that’s way too much, so shrink it down: 
#eliminate words which appear in less than 5 SMS messages.This would reduce the feature-set to a far more manageable number.
freq_terms = findFreqTerms(dtm.train, 5)
reduced_dtm.train = DocumentTermMatrix(corpus.train, list(dictionary=freq_terms))
reduced_dtm.test =  DocumentTermMatrix(corpus.test, list(dictionary=freq_terms))

# checked reduced the number of features
ncol(reduced_dtm.train)
ncol(reduced_dtm.test)
```
```r
#since NB works on factors, but this DTM only has numerics. 
#so define a function which converts counts to yes/no 
convert_counts = function(x) {
  x = ifelse(x > 0, 1, 0)
  x = factor(x, levels = c(0, 1), labels=c("No", "Yes"))
  return (x)
}

# apply() allows to work either with rows or columns of a matrix.
# MARGIN = 1 is for rows, and 2 for columns
reduced_dtm.train = apply(reduced_dtm.train, MARGIN=2, convert_counts)
reduced_dtm.test  = apply(reduced_dtm.test, MARGIN=2, convert_counts)
```
```r
#install e1071 for NB
install.packages("e1071")
#Training the model and using it for classification is a 2-stage job. 
#call naiveBayes() for running the model, then predict()
library("e1071")
# storing the model in sms_classifier
sms_classifier = naiveBayes(reduced_dtm.train, sms.train$type)
sms_test.predicted = predict(sms_classifier,
                             reduced_dtm.test)
table(sms_test.predicted, sms.test$type) #confusion matrix for spam and ham on actual and predicted data
```
```r
# checking the accuracy
#install caret for confusionmatrix
install.packages("caret")
library("caret")
confusionMatrix(table(sms_test.predicted, sms.test$type)) #check the result
```







