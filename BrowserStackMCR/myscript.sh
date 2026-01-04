#!/bin/bash

fruits=(apples oranges mangoes jackfruit banana)

for i in {1..5000}; do
  timestamp=$(date)
  quantity=$((RANDOM % 50 + 1))                     # Random quantity (1 to 50)
  price=$((RANDOM % 20 + 1))                        # Random price in $
  fruit=${fruits[$RANDOM % ${#fruits[@]}]}          # Random fruit
  echo "[$timestamp] $quantity $fruit sold for ${price}$" >> ./sample.log
  sleep_time=$(awk -v seed=$RANDOM 'BEGIN { srand(seed); printf("%.3f", rand()) }')
  sleep $sleep_time
done