* Date:   03/02/2024
* Title:  Language Information Density
* Author: Pedro Aceves

clear all
set more off
use ratio_words

la var information_density "Language Information Density (Huffman Size)"
la var ratio_words         "Language Information Density (Word Count)"

* Fig. 1a
hist information_density, graphregion(color(white)) bgcolor(white) color(navy)
* Fig. 1b
hist ratio_words, graphregion(color(white)) bgcolor(white) color(navy)
* Fig. 1c
scatter ratio_words information_density, graphregion(color(white)) bgcolor(white)

// Main model of the paper but using the corrected logged corpus size variable
mixed semantic_density information_density corpus_size_log i.corpus || fam: || language_cat: , covariance(unstructured) vce(robust)
*Fig. 2a
margins, at(c.information_density=(-230(20)-60))
marginsplot, recast(line) recastci(rarea) graphregion(color(white)) bgcolor(white) ytitle("Predicted Conceptual Space Density")

// Replicate the findings with the ratio of word count variable, which takes into account the varying size of documents within a corpus
mixed semantic_density ratio_words corpus_size_log i.corpus || fam: || language_cat: , covariance(unstructured) vce(robust)
* Fig. 2b
margins, at(c.ratio_words=(-250(20)-40))
marginsplot, recast(line) recastci(rarea) graphregion(color(white)) bgcolor(white) ytitle("Predicted Conceptual Space Density")

