# :bar_chart: Data-Science-Projects

### Repository made for some Data Science projects made internally in the Data Science reasearch area of [@TuringUSP](https://github.com/turing-usp), with the purpose of empowering its members' hard skills.
---

## :star2:		 1 - Astronomy with Pipeline

The simplest task of all these projects. The purpose of this one was basically get acquainted with Pipeline processes in Machile Learning. I used it in a model that classifies Stars, using the [Stellar Classification Dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17).

## :robot:		 2 - ML Models with SHAP

The goal of this project was to use some tools to explain and interpret a Machine Learning model, paying some special attention to the SHAP library.

For this task, I decided to train two models. The first one was a [Fetal Health Dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification), whose data indicates a fetal health status: normal, suspect, pathological. The second model was a [Bank Note Authentication Dataset](https://www.kaggle.com/datasets/ritesaluja/bank-note-authentication-uci-data), whose data points out if a bank note is legit or false.

In both models, I tried to use some libraries for tree classification models visualization, plot some decision boundaries and feature importance with SHAP to try to understand better the models I made.

## :zap:	 3 - PokeTuring with Streamlit

The idea of this project was to apply some of Streamlit tools in a web application. I decided to use it on a [Pokemon dataset](https://www.kaggle.com/datasets/rounakbanik/pokemon).

Then, running __poke.py__ you'll see the result of my efforts to visualize some cool information about Pokemons and try to understad some factors which might
affect if they're legendary or not, to create a kind of a Pokedex and, at last, to predict if a Pokemon is legendary or not based on some of its aspects.

It may be a little bit confusing for those who're not familiar with Pokemon world, so I suggest you check the Kaggle webwsite where I've found the dataset 
(so you can understand it better). But, anyway, I'll try to explain the main stats I play with using Streamlit, so you don't get lost:

* base_egg_steps: the number of steps necessary to hatch your Pokemon's egg
* base_happiness: the measure of your Pokemon's happiness
* base_total: the total amount of your Pokemon's base statistics
* sp_attack: the measure for your Pokemon's special attack
* sp_defense: the measure for your Pokemon's special attack
* pokedex: a dictionary of Pokemons. It lists all details about each living and known Pokemon
* pokedex_number: the position a Pokemon is in the Pokedex
* capture_rate: it measures how easily a Pokemon can be captured. Pokemons with high capture rates are easy to catch. On the other hand, Pokemons with low capture rates are hard to catch
* generation: each Pokemon belongs to a generation and each generation has its own Pokemons. It's like each franchise of Pokemon games/shows/etc.
* types: it is a measure to classify and aggregate Pokemons according to their behaviour. All Pokemons have at leats one type and some have two types. It can be fire, water, dragon, ghost, fairy, etc.
* against_type: it indicates how strong a Pokemon is against a certain type of Pokemon. Examples: against_fairy suggests how strong a Pokemon is against fairy Pokemons, etc.
* abilities: it indicates your Pokemon skills and abilities, which can be used in a battle
* experience_growth: it indicates how much a Pokemon grows a level due to its experiences in battles
* hp (hit points): it measures your Pokemon health
* speed: your Pokemon velocity
* percentage_male: it indicates the ratio of male Pokemons in a Pokemon species. 
