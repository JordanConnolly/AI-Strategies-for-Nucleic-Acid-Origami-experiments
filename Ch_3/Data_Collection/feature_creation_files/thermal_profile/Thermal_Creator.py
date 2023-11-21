"""Combines all of the Temperature Time Series creators
into callable functions"""

import pandas as pd


def incubated(peak, rate):
    """
    Use this for incubated temperature time series,
    where inputs are:
    Rate: Seconds, Peak: Incubation Temperature
    """
    # Lists
    time_list = []
    number_list = []
    # Loop to create list variables
    for i in range(int(rate)):
        time_list.append(i + 1)
        numbers = peak
        number_list.append(numbers)
    series = pd.Series(number_list)
    # list required
    return series


def linear(peak, base, rate):
    """
    Use this for linear temperature time series,
    where inputs are:
    Rate: Degrees Celcius change/sec, Peak: Start Temp., Base: End Temp.
    """
    # Calculations
    difference = peak - base
    seconds = difference / rate
    seconds = round(seconds, 1)
    print("seconds:", seconds)

    # Lists
    time_list = []
    number_list = []

    # Loop to create list variables
    for i in range(int(seconds)):
        time_list.append(i + 1)
        numbers = (peak - (rate * i))
        number_list.append(round(numbers, 3))

    series = pd.Series(number_list)
    # list required
    return series


def stepwise(peak, base, rate, modulo):
    """""
    Use this for linear temperature time series,
    Modulo: if it is 1 degree every 10 seconds, modulo = 10,
    Rate: Degrees celsius change/sec, Peak: Start Temp., Base: End Temp. 
    """""
    # Calculations
    difference = peak - base
    if rate % 1 != 0:
        number_divisible = difference / rate
        seconds = number_divisible * modulo
        # print("type: 1", "stats:", peak, base, rate, modulo,
        #       "\n", "rate:", rate, "divisible:", number_divisible)
        # print(seconds, "\n")
        seconds = round(seconds, 1)
    else:
        number_divisible = difference/rate
        seconds = number_divisible * modulo
        # print("type: 2", "stats:", peak, base, rate, modulo,
        #      "\n", "rate:", rate, "divisible:", number_divisible)
        # print(seconds, "\n")
        seconds = round(seconds, 1)
    # Lists for time, temperature values
    time_list = []
    number_list = []
    temp = peak
    # Loop to create list variables
    count = 0
    for j in range(int(seconds)):
        if (j + 1) % modulo == 0:
            count += 1
            # print("temp change:", count)
            temp -= rate
            # print("temp:", temp)
            rounded_temp = round(temp, 3)
            number_list.append(rounded_temp)
            time_list.append(j + 1)
        else:
            time_list.append(j + 1)
            rounded_temp = round(temp, 3)
            number_list.append(rounded_temp)
    series = pd.Series(number_list)
    # list required
    return series
