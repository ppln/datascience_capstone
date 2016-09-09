library(shiny)
library(tm)
library(RWeka)


freq_1_final <- get(load('./data/freq_1_final.RData'))
freq_2_final <- get(load('./data/freq_2_final.RData'))
freq_3_final <- get(load('./data/freq_3_final.RData'))
freq_4_final <- get(load('./data/freq_4_final.RData'))

cleanInputString <- function(input){
        
        input <- "The guy in the front of me just bought a pound of bacon, a bouquet, and a case of"
        
        # build corpus
        corpus <- Corpus(VectorSource(input))
        
        # clean data
        corpus <- tm_map(corpus, removePunctuation)
        corpus <- tm_map(corpus, content_transformer(tolower))
        corpus <- tm_map(corpus, stripWhitespace)
        corpus <- tm_map(corpus, removeNumbers)
        corpus <- tm_map(corpus, removeWords, stopwords("english"))
        corpus <- tm_map(corpus, PlainTextDocument)
        
        words <- unlist(strsplit(corpus[[1]]$content, ' +'))
        
        return(words)
}

shinyServer(function(input, output){
        
        words <- cleanInputString(input)
        len_words <- length(words)
        if(len_words > 3){
                sub_str <- paste(words[(len_words-2):len_words], collapse = ' ')
                
                idx <- grep(paste('^', sub_str, sep = ''), freq_4_final$word)
                
                if(nrow(freq_4_final[idx, ]) > 0){
                        predict_string <- freq_4_final[idx, ][1,1]
                        predict_string <- strsplit(predict_string, ' ')[[1]]
                        predict_string[length(predict_string)]
                }
                        
        }
                
        
        
        
})