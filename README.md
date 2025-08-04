# League of Legends Main Recommender

Suggests champions to play in LoL for a given summoner. Meant to be more of an educational project, comparing results of different recommendation systems.

## Current Features
ALS Model-Based Collaborative Filter
SGD Model-Based Collaborative Filter
Content Based Filter
Simple Hybrid Filter

## Overview
Uses some permutation of summoner history and champion metadata to come up with scores that recommend a summoner's estimated proclivity for a given champion. Data is generated starting from the top players' ranked histories and then snowballed based on other users in the matches. Challenger players are chosen as a representative sample of player archetypes and optimal play, avoiding too many wacky selections in lower elos. 

## Libraries
Pytorch