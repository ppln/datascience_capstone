setwd('I:/coursera/10_Capstone/git_project/datascience_capstone')

# load libraries
library(tm)
library(RWeka)
library(stringr)

# load raw data

twitters <- readLines('./Coursera-SwiftKey/final/en_US/en_US.twitter.txt', encoding = 'UTF-8', skipNul = TRUE)
blogs <- readLines('./Coursera-SwiftKey/final/en_US/en_US.blogs.txt', encoding = 'UTF-8', skipNul = TRUE)

news_file <- file('./Coursera-SwiftKey/final/en_US/en_US.news.txt')
news <- readLines(news_file, encoding = 'UTF-8', skipNul = FALSE)
close(news_file)

# data sampling
sample_percent <- 0.20
set.seed(2213)
sub_twitter <- sample(twitters, length(twitters) * sample_percent)
sub_blogs <- sample(blogs, length(blogs) * sample_percent)
sub_news <- sample(news, length(news) * sample_percent)

# merge the three data sets into a larger one
total <- c(sub_twitter, sub_blogs, sub_news)

# convert invalid characters
total <- iconv(total, "latin1", "ASCII", sub = '')

# remove some variables to save memory
rm(twitters, blogs, news, sub_twitter, sub_blogs, sub_news)
#gc(reset = TRUE)


step_one <- function(total){
        
        loop <- 100
        size <- as.integer(length(total)/loop)
        
        for(i in 1:loop){
                
                min_idx <- (i - 1) * size + 1
                max_idx <- i * size
                
                sub_data <- total[min_idx : max_idx]
                
                # build corpus
                corpus <- Corpus(VectorSource(sub_data))
                
                # clean data
                corpus <- tm_map(corpus, removePunctuation)
                corpus <- tm_map(corpus, content_transformer(tolower))
                corpus <- tm_map(corpus, stripWhitespace)
                corpus <- tm_map(corpus, removeNumbers)
                #corpus <- tm_map(corpus, removeWords, stopwords("english"))
                corpus <- tm_map(corpus, PlainTextDocument)
                
                # Tokenization N-gram
                for(n in 1:4){
                        
                        # Tokenization N-gram
                        gram <- function(x) NGramTokenizer(x, Weka_control(min = n, max = n))
                        
                        # create Term-Document Matrices
                        tdm <- TermDocumentMatrix(corpus, control = list(tokenize = gram))
                        
                        # calculate the frequency
                        freq <- sort(rowSums(as.matrix(tdm)), decreasing = TRUE)
                        
                        # build data frame
                        freq_data <- data.frame(word = names(freq), frequency = freq)
                        
                        # write part data into file
                        file_name <- paste('./processingdata/freq_', n, sep = '')
                        write.table(freq_data, file_name, append = TRUE, 
                                    col.names = FALSE, row.names = FALSE)
                }
                
                print(i)
        }
}


step_two <- function(){
        for(n in 1:4){
                
                file_name <- paste('./processingdata/freq_', n, sep = '')
                freq_data <- read.table(file_name, header = FALSE, stringsAsFactors = FALSE)
                
                colnames(freq_data) <- c('word', 'frequency')
                freq_data <- aggregate(frequency ~ word, data = freq_data, FUN = sum)
                
                freq_data <- freq_data[freq_data$frequency > 1, ]
                
                # sort
                freq_data <- freq_data[order(-freq_data$frequency), ]
                
                save(freq_data, file = paste(file_name, '_final.RData', sep = ''))
                
                write.table(freq_data, paste(file_name, '_final.txt', sep = ''), 
                            col.names = TRUE, row.names = FALSE)
        }
}


step_one(total)

step_two()

