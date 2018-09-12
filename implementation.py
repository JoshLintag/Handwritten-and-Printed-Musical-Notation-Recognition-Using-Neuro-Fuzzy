#handwriting deficiency detection

import numpy as np
import skfuzzy as fz
import matplotlib.pyplot as plt
import pandas as pd
import csv
from tkinter.filedialog import askopenfilename

###################################METHODS#####################################
#gets the mean of each column (raw data of each feature)
def getMean(numbers):
    return sum(numbers)/float(len(numbers))

#Putting the data in a .csv file
def createCSV(list1, list2, list3, list4, list5, list6):
    with open('output.csv', 'w') as myFile:
        wr = csv.writer(myFile)
        wr.writerow(list1)
        wr.writerow(list2)
        wr.writerow(list3)     
        wr.writerow(list4)
        wr.writerow(list5)
        wr.writerow(list6)

#Fuzzified Data to Membership functions
def feature_pressure(in_pressure):
    lvl_pressure_lo = fz.interp_membership(pressure, pressure_lo, in_pressure)
    lvl_pressure_md = fz.interp_membership(pressure, pressure_md, in_pressure)
    lvl_pressure_hi = fz.interp_membership(pressure, pressure_hi, in_pressure)
    return dict(hard = lvl_pressure_hi, moderate = lvl_pressure_md, light = lvl_pressure_lo)

def feature_slant(in_slant):
    lvl_slant_lo = fz.interp_membership(slant, slant_lo, in_slant)
    lvl_slant_md = fz.interp_membership(slant, slant_md, in_slant)
    lvl_slant_hi = fz.interp_membership(slant, slant_hi, in_slant)
    return dict(negative = lvl_slant_lo, vertical = lvl_slant_md, positive = lvl_slant_hi)

def feature_penDown(in_penDown):
    lvl_pen_lo = fz.interp_membership(penDown, penDown_lo, in_penDown)
    lvl_pen_md = fz.interp_membership(penDown, penDown_md, in_penDown)
    lvl_pen_hi = fz.interp_membership(penDown, penDown_hi, in_penDown)
    return dict(short = lvl_pen_lo, average = lvl_pen_md, long = lvl_pen_hi)

def feature_duration(in_duration):
    lvl_duration_lo = fz.interp_membership(duration, duration_lo, in_duration)
    lvl_duration_md = fz.interp_membership(duration, duration_md, in_duration)
    lvl_duration_hi = fz.interp_membership(duration, duration_hi, in_duration)
    return dict(short = lvl_duration_lo, average = lvl_duration_md, long = lvl_duration_hi)

def feature_velocity(in_velocity):
    lvl_velocity_lo = fz.interp_membership(velocity, velocity_lo, in_velocity)
    lvl_velocity_md = fz.interp_membership(velocity, velocity_md, in_velocity)
    lvl_velocity_hi = fz.interp_membership(velocity, velocity_hi, in_velocity)
    return dict(slow = lvl_velocity_lo, moderate = lvl_velocity_md, fast = lvl_velocity_hi)

def feature_jerk(in_jerk):
    lvl_jerk_lo = fz.interp_membership(jerk, jerk_lo, in_jerk)
    lvl_jerk_md = fz.interp_membership(jerk, jerk_md, in_jerk)
    lvl_jerk_hi = fz.interp_membership(jerk, jerk_hi, in_jerk)
    return dict(low = lvl_jerk_lo, medium = lvl_jerk_md, high = lvl_jerk_hi)

#Aggregation of all outputs
def aggregate(activation1, activation2, activation3, num):    
    aggregated = np.fmax(activation1, np.fmax(activation2, activation3))
    defuzzify(aggregated, num)

#defuzzification
def defuzzify(aggregate_membership, num):    
    result_classification = fz.defuzz(score, aggregate_membership, 'centroid')
    print (result_classification)
    
    num.append(result_classification)
    plotting = fz.interp_membership(score, aggregate_membership, result_classification)
    result(plotting, aggregate_membership, result_classification)

#plotting the precise classification line    
def result(result_plot, aggregate, res_class):
    score0 = np.zeros_like(score)
    fig, res_plot = plt.subplots(figsize=(8, 3))
    
    res_plot.plot(score, score_lo, 'b', linewidth=0.5, linestyle='--', )
    res_plot.plot(score, score_md, 'g', linewidth=0.5, linestyle='--')
    res_plot.plot(score, score_hi, 'r', linewidth=0.5, linestyle='--')
    res_plot.fill_between(score, score0, aggregate, facecolor='Pink', alpha=0.7)
    res_plot.plot([res_class, res_class], [0, result_plot], 'k', linewidth=1.5, alpha=0.9)
    res_plot.set_title('Precise Classification Result (line)')
    
    for res in (res_plot, ):
        res.spines['top'].set_visible(False)
        res.spines['right'].set_visible(False)
        res.get_xaxis().tick_bottom()
        res.get_yaxis().tick_left()
        
    plt.tight_layout()

#Rules Definition--1 variable
#####################################START#####################################

#IF pressure
def pr():
    global one
    rule1_pr = in_pressure['hard']
    rule2_pr = in_pressure['moderate']
    rule3_pr = in_pressure['light']
    #THEN
    activation_pr_poor = np.fmin(rule1_pr, score_hi)
    activation_pr_ave = np.fmin(rule2_pr, score_md)
    activation_pr_prof = np.fmin(rule3_pr, score_lo)
    
    print ('pr: ')    
    aggregate(activation_pr_poor, activation_pr_ave, activation_pr_prof, one)

#slant
def slt():
    global one
    rule1_slt = in_slant['negative']
    rule2_slt = in_slant['positive']
    rule3_slt = in_slant['vertical']
    #THEN
    activation_slt_poor = np.fmin(rule1_slt, score_hi)
    activation_slt_ave = np.fmin(rule2_slt, score_md)
    activation_slt_prof = np.fmin(rule3_slt, score_lo)
    
    print ('slt: ')    
    aggregate(activation_slt_poor, activation_slt_ave, activation_slt_prof, one)

#pendown 
def pen():
    global one
    rule1_pen = in_penDown['short']
    rule2_pen = in_penDown['average']
    rule3_pen = in_penDown['long']
    #THEN
    activation_pen_poor = np.fmin(rule1_pen, score_hi)
    activation_pen_ave = np.fmin(rule2_pen, score_md)
    activation_pen_prof = np.fmin(rule3_pen, score_lo)
    
    print ('pen: ')    
    aggregate(activation_pen_poor, activation_pen_ave, activation_pen_prof, one)
    
#duration
def dur():
    global one
    rule1_dur = in_duration['long']
    rule2_dur = in_duration['average']
    rule3_dur = in_duration['short']
    #THEN
    activation_dur_poor = np.fmin(rule1_dur, score_hi)
    activation_dur_ave = np.fmin(rule2_dur, score_md)
    activation_dur_prof = np.fmin(rule3_dur, score_lo)
    
    print ('dur: ')    
    aggregate(activation_dur_poor, activation_dur_ave, activation_dur_prof, one)
    
def vel():
    global one
    rule1_vel = in_velocity['slow']
    rule2_vel = in_velocity['moderate']
    rule3_vel = in_velocity['fast']
    #THEN
    activation_vel_poor = np.fmin(rule1_vel, score_hi)
    activation_vel_ave = np.fmin(rule2_vel, score_md)
    activation_vel_prof = np.fmin(rule3_vel, score_lo)
    
    print ('vel: ')    
    aggregate(activation_vel_poor, activation_vel_ave, activation_vel_prof, one)

#pressure and jerk
def jer():
    global one
    rule1_jer = in_jerk['high']
    rule2_jer = in_jerk['medium']
    rule3_jer = in_jerk['low']
    #THEN
    activation_jer_poor = np.fmin(rule1_jer, score_hi)
    activation_jer_ave = np.fmin(rule2_jer, score_md)
    activation_jer_prof = np.fmin(rule3_jer, score_lo)
    
    print ('jer: ')    
    aggregate(activation_jer_poor, activation_jer_ave, activation_jer_prof, one)    
    
######################################END######################################

#Rules Definition--2 variables
#####################################START#####################################

#IF pressure and slant
def pr_slt():
    global two
    rule1_pr_slt = np.fmax(in_pressure['hard'], in_slant['negative'])
    rule2_pr_slt = in_pressure['moderate']
    rule3_pr_slt = np.fmax(in_pressure['light'], in_slant['vertical'])
    #THEN
    activation_pr_slt_poor = np.fmin(rule1_pr_slt, score_hi)
    activation_pr_slt_ave = np.fmin(rule2_pr_slt, score_md)
    activation_pr_slt_prof = np.fmin(rule3_pr_slt, score_lo)
    
    print ('pr_slt: ')    
    aggregate(activation_pr_slt_poor, activation_pr_slt_ave, activation_pr_slt_prof, two)
    
#pressure and pen down
def pr_pen():
    global two
    rule1_pr_pen = np.fmax(in_pressure['hard'], in_penDown['short'])
    rule2_pr_pen = np.fmax(in_pressure['moderate'], in_penDown['average'])
    rule3_pr_pen = np.fmax(in_pressure['light'], in_penDown['long'])
    #THEN
    activation_pr_pen_poor = np.fmin(rule1_pr_pen, score_hi)
    activation_pr_pen_ave = np.fmin(rule2_pr_pen, score_md)
    activation_pr_pen_prof = np.fmin(rule3_pr_pen, score_lo)
    
    print ('pr_pen: ')    
    aggregate(activation_pr_pen_poor, activation_pr_pen_ave, activation_pr_pen_prof, two)

#pressure and duration
def pr_dur():
    global two
    rule1_pr_dur = np.fmax(in_pressure['hard'], in_duration['long'])
    rule2_pr_dur = np.fmax(in_pressure['moderate'], in_duration['average'])
    rule3_pr_dur = np.fmax(in_pressure['light'], in_duration['short'])
    #THEN
    activation_pr_dur_poor = np.fmin(rule1_pr_dur, score_hi)
    activation_pr_dur_ave = np.fmin(rule2_pr_dur, score_md)
    activation_pr_dur_prof = np.fmin(rule3_pr_dur, score_lo)
    
    print ('pr_dur: ')    
    aggregate(activation_pr_dur_poor, activation_pr_dur_ave, activation_pr_dur_prof, two)

#pressure and velocity
def pr_vel():
    global two
    rule1_pr_vel = np.fmax(in_pressure['hard'], in_velocity['slow'])
    rule2_pr_vel = np.fmax(in_pressure['moderate'], in_velocity['moderate'])
    rule3_pr_vel = np.fmax(in_pressure['light'], in_velocity['fast'])
    #THEN
    activation_pr_vel_poor = np.fmin(rule1_pr_vel, score_hi)
    activation_pr_vel_ave = np.fmin(rule2_pr_vel, score_md)
    activation_pr_vel_prof = np.fmin(rule3_pr_vel, score_lo)
    
    print ('pr_vel: ')    
    aggregate(activation_pr_vel_poor, activation_pr_vel_ave, activation_pr_vel_prof, two)

#pressure and jerk
def pr_jer():
    global two
    rule1_pr_jer = np.fmax(in_pressure['hard'], in_jerk['high'])
    rule2_pr_jer = np.fmax(in_pressure['moderate'], in_jerk['medium'])
    rule3_pr_jer = np.fmax(in_pressure['light'], in_jerk['low'])
    #THEN
    activation_pr_jer_poor = np.fmin(rule1_pr_jer, score_hi)
    activation_pr_jer_ave = np.fmin(rule2_pr_jer, score_md)
    activation_pr_jer_prof = np.fmin(rule3_pr_jer, score_lo)
    
    print ('pr_jer: ')    
    aggregate(activation_pr_jer_poor, activation_pr_jer_ave, activation_pr_jer_prof, two)

#slant and pen down
def slt_pen():
    global two
    rule1_slt_pen = np.fmax(in_slant['negative'], in_penDown['short'])
    rule2_slt_pen = in_penDown['average']
    rule3_slt_pen = np.fmax(in_slant['vertical'], in_penDown['long'])
    #THEN
    activation_slt_pen_poor = np.fmin(rule1_slt_pen, score_hi)
    activation_slt_pen_ave = np.fmin(rule2_slt_pen, score_md)
    activation_slt_pen_prof = np.fmin(rule3_slt_pen, score_lo)
    
    print ('slt_pen: ')    
    aggregate(activation_slt_pen_poor, activation_slt_pen_ave, activation_slt_pen_prof, two)

#slant and duration
def slt_dur():
    global two
    rule1_slt_dur = np.fmax(in_slant['negative'], in_duration['long'])
    rule2_slt_dur = in_duration['average']
    rule3_slt_dur = np.fmax(in_slant['vertical'], in_duration['short'])
    #THEN
    activation_slt_dur_poor = np.fmin(rule1_slt_dur, score_hi)
    activation_slt_dur_ave = np.fmin(rule2_slt_dur, score_md)
    activation_slt_dur_prof = np.fmin(rule3_slt_dur, score_lo)
    
    print ('slt_dur: ')    
    aggregate(activation_slt_dur_poor, activation_slt_dur_ave, activation_slt_dur_prof, two)

#slant and velocity
def slt_vel():
    global two
    rule1_slt_vel = np.fmax(in_slant['negative'], in_velocity['slow'])
    rule2_slt_vel = in_velocity['moderate']
    rule3_slt_vel = np.fmax(in_slant['vertical'], in_velocity['fast'])
    #THEN
    activation_slt_vel_poor = np.fmin(rule1_slt_vel, score_hi)
    activation_slt_vel_ave = np.fmin(rule2_slt_vel, score_md)
    activation_slt_vel_prof = np.fmin(rule3_slt_vel, score_lo)
    
    print ('slt_vel: ')    
    aggregate(activation_slt_vel_poor, activation_slt_vel_ave, activation_slt_vel_prof, two)

#slant and jerk
def slt_jer():
    global two
    rule1_slt_jer = np.fmax(in_slant['negative'], in_jerk['high'])
    rule2_slt_jer = in_jerk['medium']
    rule3_slt_jer = np.fmax(in_slant['vertical'], in_jerk['low'])
    #THEN
    activation_slt_jer_poor = np.fmin(rule1_slt_jer, score_hi)
    activation_slt_jer_ave = np.fmin(rule2_slt_jer, score_md)
    activation_slt_jer_prof = np.fmin(rule3_slt_jer, score_lo)
    
    print ('slt_jer: ')    
    aggregate(activation_slt_jer_poor, activation_slt_jer_ave, activation_slt_jer_prof, two)

#pen down and duration
def pen_dur():
    global two
    rule1_pen_dur = np.fmax(in_penDown['short'], in_duration['long'])
    rule2_pen_dur = np.fmax(in_penDown['average'], in_duration['average'])
    rule3_pen_dur = np.fmax(in_penDown['long'], in_duration['short'])
    #THEN
    activation_pen_dur_poor = np.fmin(rule1_pen_dur, score_hi)
    activation_pen_dur_ave = np.fmin(rule2_pen_dur, score_md)
    activation_pen_dur_prof = np.fmin(rule3_pen_dur, score_lo)
    
    print ('pen_dur: ')    
    aggregate(activation_pen_dur_poor, activation_pen_dur_ave, activation_pen_dur_prof, two)

#pen down and velocity
def pen_vel():
    global two
    rule1_pen_vel = np.fmax(in_penDown['short'], in_velocity['slow'])
    rule2_pen_vel = np.fmax(in_penDown['average'], in_velocity['moderate'])
    rule3_pen_vel = np.fmax(in_penDown['long'], in_velocity['fast'])
    #THEN
    activation_pen_vel_poor = np.fmin(rule1_pen_vel, score_hi)
    activation_pen_vel_ave = np.fmin(rule2_pen_vel, score_md)
    activation_pen_vel_prof = np.fmin(rule3_pen_vel, score_lo)
    
    print ('pen_vel: ')    
    aggregate(activation_pen_vel_poor, activation_pen_vel_ave, activation_pen_vel_prof, two)

#pen down and jerk
def pen_jer():
    global two
    rule1_pen_jer = np.fmax(in_penDown['short'], in_jerk['high'])
    rule2_pen_jer = np.fmax(in_penDown['average'], in_jerk['medium'])
    rule3_pen_jer = np.fmax(in_penDown['long'], in_jerk['low'])
    #THEN
    activation_pen_jer_poor = np.fmin(rule1_pen_jer, score_hi)
    activation_pen_jer_ave = np.fmin(rule2_pen_jer, score_md)
    activation_pen_jer_prof = np.fmin(rule3_pen_jer, score_lo)
    
    print ('pen_jer: ')    
    aggregate(activation_pen_jer_poor, activation_pen_jer_ave, activation_pen_jer_prof, two)

#duration and velocity
def dur_vel():
    global two
    rule1_dur_vel = np.fmax(in_duration['long'], in_velocity['slow'])
    rule2_dur_vel = np.fmax(in_duration['average'], in_velocity['moderate'])
    rule3_dur_vel = np.fmax(in_duration['short'], in_velocity['fast'])
    #THEN
    activation_dur_vel_poor = np.fmin(rule1_dur_vel, score_hi)
    activation_dur_vel_ave = np.fmin(rule2_dur_vel, score_md)
    activation_dur_vel_prof = np.fmin(rule3_dur_vel, score_lo)
    
    print ('dur_vel: ')    
    aggregate(activation_dur_vel_poor, activation_dur_vel_ave, activation_dur_vel_prof, two)

#duration and jerk
def dur_jer():
    global two
    rule1_dur_jer = np.fmax(in_duration['long'], in_jerk['high'])
    rule2_dur_jer = np.fmax(in_duration['average'], in_jerk['medium'])
    rule3_dur_jer = np.fmax(in_duration['short'], in_jerk['low'])
    #THEN
    activation_dur_jer_poor = np.fmin(rule1_dur_jer, score_hi)
    activation_dur_jer_ave = np.fmin(rule2_dur_jer, score_md)
    activation_dur_jer_prof = np.fmin(rule3_dur_jer, score_lo)
    
    print ('dur_jer: ')    
    aggregate(activation_dur_jer_poor, activation_dur_jer_ave, activation_dur_jer_prof, two)

#velocity and jerk
def vel_jer():
    global two
    rule1_vel_jer = np.fmax(in_velocity['slow'], in_jerk['high'])
    rule2_vel_jer = np.fmax(in_velocity['moderate'], in_jerk['medium'])
    rule3_vel_jer = np.fmax(in_velocity['fast'], in_jerk['low'])
    #THEN
    activation_vel_jer_poor = np.fmin(rule1_vel_jer, score_hi)
    activation_vel_jer_ave = np.fmin(rule2_vel_jer, score_md)
    activation_vel_jer_prof = np.fmin(rule3_vel_jer, score_lo)
    
    print ('vel_jer: ')    
    aggregate(activation_vel_jer_poor, activation_vel_jer_ave, activation_vel_jer_prof, two)

######################################END######################################

#Rules Definition--3 variables
#####################################START#####################################

#IF pressure, slant and pen
def pr_slt_pen():
    global three
    rule1_pr_slt_pen = np.fmax(in_pressure['hard'], np.fmax(in_slant['negative'], in_penDown['short']))
    rule2_pr_slt_pen = np.fmax(in_pressure['moderate'], in_penDown['average'])
    rule3_pr_slt_pen = np.fmax(in_pressure['light'], np.fmax(in_slant['vertical'], in_penDown['long']))
    #THEN
    activation_pr_slt_pen_poor = np.fmin(rule1_pr_slt_pen, score_hi)
    activation_pr_slt_pen_ave = np.fmin(rule2_pr_slt_pen, score_md)
    activation_pr_slt_pen_prof = np.fmin(rule3_pr_slt_pen, score_lo)
    
    print ('pr_slt_pen: ')    
    aggregate(activation_pr_slt_pen_poor, activation_pr_slt_pen_ave, activation_pr_slt_pen_prof, three)

#pressure, slant and duration
def pr_slt_dur():
    global three
    rule1_pr_slt_dur = np.fmax(in_pressure['hard'], np.fmax(in_slant['negative'], in_duration['long']))
    rule2_pr_slt_dur = np.fmax(in_pressure['moderate'], in_duration['average'])
    rule3_pr_slt_dur = np.fmax(in_pressure['light'], np.fmax(in_slant['vertical'], in_duration['short']))
    #THEN
    activation_pr_slt_dur_poor = np.fmin(rule1_pr_slt_dur, score_hi)
    activation_pr_slt_dur_ave = np.fmin(rule2_pr_slt_dur, score_md)
    activation_pr_slt_dur_prof = np.fmin(rule3_pr_slt_dur, score_lo)
    
    print ('pr_slt_dur: ')    
    aggregate(activation_pr_slt_dur_poor, activation_pr_slt_dur_ave, activation_pr_slt_dur_prof, three)

#pressure, slant and velocity
def pr_slt_vel():
    global three
    rule1_pr_slt_vel = np.fmax(in_pressure['hard'], np.fmax(in_slant['negative'], in_velocity['slow']))
    rule2_pr_slt_vel = np.fmax(in_pressure['moderate'], in_velocity['moderate'])
    rule3_pr_slt_vel = np.fmax(in_pressure['light'], np.fmax(in_slant['vertical'], in_velocity['fast']))
    #THEN
    activation_pr_slt_vel_poor = np.fmin(rule1_pr_slt_vel, score_hi)
    activation_pr_slt_vel_ave = np.fmin(rule2_pr_slt_vel, score_md)
    activation_pr_slt_vel_prof = np.fmin(rule3_pr_slt_vel, score_lo)
    
    print ('pr_slt_vel: ')    
    aggregate(activation_pr_slt_vel_poor, activation_pr_slt_vel_ave, activation_pr_slt_vel_prof, three)

#pressure, slant and jerk
def pr_slt_jer():
    global three
    rule1_pr_slt_jer = np.fmax(in_pressure['hard'], np.fmax(in_slant['negative'], in_jerk['high']))
    rule2_pr_slt_jer = np.fmax(in_pressure['moderate'], in_jerk['medium'])
    rule3_pr_slt_jer = np.fmax(in_pressure['light'], np.fmax(in_slant['vertical'], in_jerk['low']))
    #THEN
    activation_pr_slt_jer_poor = np.fmin(rule1_pr_slt_jer, score_hi)
    activation_pr_slt_jer_ave = np.fmin(rule2_pr_slt_jer, score_md)
    activation_pr_slt_jer_prof = np.fmin(rule3_pr_slt_jer, score_lo)
    
    print ('pr_slt_jer: ')    
    aggregate(activation_pr_slt_jer_poor, activation_pr_slt_jer_ave, activation_pr_slt_jer_prof, three)

#pressure, pen down and duration
def pr_pen_dur():
    global three
    rule1_pr_pen_dur = np.fmax(in_pressure['hard'], np.fmax(in_penDown['short'], in_duration['long']))
    rule2_pr_pen_dur = np.fmax(in_pressure['moderate'], np.fmax(in_penDown['average'], in_duration['average']))
    rule3_pr_pen_dur = np.fmax(in_pressure['light'], np.fmax(in_penDown['long'], in_duration['short']))
    #THEN
    activation_pr_pen_dur_poor = np.fmin(rule1_pr_pen_dur, score_hi)
    activation_pr_pen_dur_ave = np.fmin(rule2_pr_pen_dur, score_md)
    activation_pr_pen_dur_prof = np.fmin(rule3_pr_pen_dur, score_lo)
    
    print ('pr_pen_dur: ')    
    aggregate(activation_pr_pen_dur_poor, activation_pr_pen_dur_ave, activation_pr_pen_dur_prof, three)

#pressure, pen down and velocity
def pr_pen_vel():
    global three
    rule1_pr_pen_vel = np.fmax(in_pressure['hard'], np.fmax(in_penDown['short'], in_velocity['slow']))
    rule2_pr_pen_vel = np.fmax(in_pressure['moderate'], np.fmax(in_penDown['average'], in_velocity['moderate']))
    rule3_pr_pen_vel = np.fmax(in_pressure['light'], np.fmax(in_penDown['long'], in_velocity['fast']))
    #THEN
    activation_pr_pen_vel_poor = np.fmin(rule1_pr_pen_vel, score_hi)
    activation_pr_pen_vel_ave = np.fmin(rule2_pr_pen_vel, score_md)
    activation_pr_pen_vel_prof = np.fmin(rule3_pr_pen_vel, score_lo)
    
    print ('pr_pen_vel: ')    
    aggregate(activation_pr_pen_vel_poor, activation_pr_pen_vel_ave, activation_pr_pen_vel_prof, three)

#pressure, pen down and jerk
def pr_pen_jer():
    global three
    rule1_pr_pen_jer = np.fmax(in_pressure['hard'], np.fmax(in_penDown['short'], in_jerk['high']))
    rule2_pr_pen_jer = np.fmax(in_pressure['moderate'], np.fmax(in_penDown['average'], in_jerk['medium']))
    rule3_pr_pen_jer = np.fmax(in_pressure['light'], np.fmax(in_penDown['long'], in_jerk['low']))
    #THEN
    activation_pr_pen_jer_poor = np.fmin(rule1_pr_pen_jer, score_hi)
    activation_pr_pen_jer_ave = np.fmin(rule2_pr_pen_jer, score_md)
    activation_pr_pen_jer_prof = np.fmin(rule3_pr_pen_jer, score_lo)
    
    print ('pr_pen_jer: ')    
    aggregate(activation_pr_pen_jer_poor, activation_pr_pen_jer_ave, activation_pr_pen_jer_prof, three)

#pressure, duration and velocity
def pr_dur_vel():
    global three
    rule1_pr_dur_vel = np.fmax(in_pressure['hard'], np.fmax(in_duration['long'], in_velocity['slow']))
    rule2_pr_dur_vel = np.fmax(in_pressure['moderate'], np.fmax(in_duration['average'], in_velocity['moderate']))
    rule3_pr_dur_vel = np.fmax(in_pressure['light'], np.fmax(in_duration['short'], in_velocity['fast']))
    #THEN
    activation_pr_dur_vel_poor = np.fmin(rule1_pr_dur_vel, score_hi)
    activation_pr_dur_vel_ave = np.fmin(rule2_pr_dur_vel, score_md)
    activation_pr_dur_vel_prof = np.fmin(rule3_pr_dur_vel, score_lo)
    
    print ('pr_dur_vel: ')    
    aggregate(activation_pr_dur_vel_poor, activation_pr_dur_vel_ave, activation_pr_dur_vel_prof, three)

#pressure, duration and jerk
def pr_dur_jer():
    global three
    rule1_pr_dur_jer = np.fmax(in_pressure['hard'], np.fmax(in_duration['long'], in_jerk['high']))
    rule2_pr_dur_jer = np.fmax(in_pressure['moderate'], np.fmax(in_duration['average'], in_jerk['medium']))
    rule3_pr_dur_jer = np.fmax(in_pressure['light'], np.fmax(in_duration['short'], in_jerk['low']))
    #THEN
    activation_pr_dur_jer_poor = np.fmin(rule1_pr_dur_jer, score_hi)
    activation_pr_dur_jer_ave = np.fmin(rule2_pr_dur_jer, score_md)
    activation_pr_dur_jer_prof = np.fmin(rule3_pr_dur_jer, score_lo)
    
    print ('pr_dur_jer: ')    
    aggregate(activation_pr_dur_jer_poor, activation_pr_dur_jer_ave, activation_pr_dur_jer_prof, three)

#pressure, velocity and jerk
def pr_vel_jer():
    global three
    rule1_pr_vel_jer = np.fmax(in_pressure['hard'], np.fmax(in_velocity['slow'], in_jerk['high']))
    rule2_pr_vel_jer = np.fmax(in_pressure['moderate'], np.fmax(in_velocity['moderate'], in_jerk['medium']))
    rule3_pr_vel_jer = np.fmax(in_pressure['light'], np.fmax(in_velocity['fast'], in_jerk['low']))
    #THEN
    activation_pr_vel_jer_poor = np.fmin(rule1_pr_vel_jer, score_hi)
    activation_pr_vel_jer_ave = np.fmin(rule2_pr_vel_jer, score_md)
    activation_pr_vel_jer_prof = np.fmin(rule3_pr_vel_jer, score_lo)
    
    print ('pr_vel_jer: ')    
    aggregate(activation_pr_vel_jer_poor, activation_pr_vel_jer_ave, activation_pr_vel_jer_prof, three)

#slant, pen down and duration
def slt_pen_dur():
    global three
    rule1_slt_pen_dur = np.fmax(in_slant['negative'], np.fmax(in_penDown['short'], in_duration['long']))
    rule2_slt_pen_dur = np.fmax(in_penDown['average'], in_duration['average'])
    rule3_slt_pen_dur = np.fmax(in_slant['vertical'], np.fmax(in_penDown['long'], in_duration['short']))
    #THEN
    activation_slt_pen_dur_poor = np.fmin(rule1_slt_pen_dur, score_hi)
    activation_slt_pen_dur_ave = np.fmin(rule2_slt_pen_dur, score_md)
    activation_slt_pen_dur_prof = np.fmin(rule3_slt_pen_dur, score_lo)
    
    print ('slt_pen_dur: ')    
    aggregate(activation_slt_pen_dur_poor, activation_slt_pen_dur_ave, activation_slt_pen_dur_prof, three)

#slant, pen down and velocity
def slt_pen_vel():
    global three
    rule1_slt_pen_vel = np.fmax(in_slant['negative'], np.fmax(in_penDown['short'], in_velocity['slow']))
    rule2_slt_pen_vel = np.fmax(in_penDown['average'], in_velocity['moderate'])
    rule3_slt_pen_vel = np.fmax(in_slant['vertical'], np.fmax(in_penDown['long'], in_velocity['fast']))
    #THEN
    activation_slt_pen_vel_poor = np.fmin(rule1_slt_pen_vel, score_hi)
    activation_slt_pen_vel_ave = np.fmin(rule2_slt_pen_vel, score_md)
    activation_slt_pen_vel_prof = np.fmin(rule3_slt_pen_vel, score_lo)
    
    print ('slt_pen_vel: ')    
    aggregate(activation_slt_pen_vel_poor, activation_slt_pen_vel_ave, activation_slt_pen_vel_prof, three)

#slant, pen down and jerk
def slt_pen_jer():
    global three
    rule1_slt_pen_jer = np.fmax(in_slant['negative'], np.fmax(in_penDown['short'], in_jerk['high']))
    rule2_slt_pen_jer = np.fmax(in_penDown['average'], in_jerk['medium'])
    rule3_slt_pen_jer = np.fmax(in_slant['vertical'], np.fmax(in_penDown['long'], in_jerk['low']))
    #THEN
    activation_slt_pen_jer_poor = np.fmin(rule1_slt_pen_jer, score_hi)
    activation_slt_pen_jer_ave = np.fmin(rule2_slt_pen_jer, score_md)
    activation_slt_pen_jer_prof = np.fmin(rule3_slt_pen_jer, score_lo)
    
    print ('slt_pen_jer: ')    
    aggregate(activation_slt_pen_jer_poor, activation_slt_pen_jer_ave, activation_slt_pen_jer_prof, three)

#slant, duration and velocity
def slt_dur_vel():
    global three
    rule1_slt_dur_vel = np.fmax(in_slant['negative'], np.fmax(in_duration['long'], in_velocity['slow']))
    rule2_slt_dur_vel = np.fmax(in_duration['average'], in_velocity['moderate'])
    rule3_slt_dur_vel = np.fmax(in_slant['vertical'], np.fmax(in_duration['short'], in_velocity['fast']))
    #THEN
    activation_slt_dur_vel_poor = np.fmin(rule1_slt_dur_vel, score_hi)
    activation_slt_dur_vel_ave = np.fmin(rule2_slt_dur_vel, score_md)
    activation_slt_dur_vel_prof = np.fmin(rule3_slt_dur_vel, score_lo)
    
    print ('slt_dur_vel: ')    
    aggregate(activation_slt_dur_vel_poor, activation_slt_dur_vel_ave, activation_slt_dur_vel_prof, three)

#slant, duration and jerk
def slt_dur_jer():
    global three
    rule1_slt_dur_jer = np.fmax(in_slant['negative'], np.fmax(in_duration['long'], in_jerk['high']))
    rule2_slt_dur_jer = np.fmax(in_duration['average'], in_jerk['medium'])
    rule3_slt_dur_jer = np.fmax(in_slant['vertical'], np.fmax(in_duration['short'], in_jerk['low']))
    #THEN
    activation_slt_dur_jer_poor = np.fmin(rule1_slt_dur_jer, score_hi)
    activation_slt_dur_jer_ave = np.fmin(rule2_slt_dur_jer, score_md)
    activation_slt_dur_jer_prof = np.fmin(rule3_slt_dur_jer, score_lo)
    
    print ('slt_dur_jer: ')    
    aggregate(activation_slt_dur_jer_poor, activation_slt_dur_jer_ave, activation_slt_dur_jer_prof, three)

#slant, velocity and jerk
def slt_vel_jer():
    global three
    rule1_slt_vel_jer = np.fmax(in_slant['negative'], np.fmax(in_velocity['slow'], in_jerk['high']))
    rule2_slt_vel_jer = np.fmax(in_velocity['moderate'], in_jerk['medium'])
    rule3_slt_vel_jer = np.fmax(in_slant['vertical'], np.fmax(in_velocity['fast'], in_jerk['low']))
    #THEN
    activation_slt_vel_jer_poor = np.fmin(rule1_slt_vel_jer, score_hi)
    activation_slt_vel_jer_ave = np.fmin(rule2_slt_vel_jer, score_md)
    activation_slt_vel_jer_prof = np.fmin(rule3_slt_vel_jer, score_lo)
    
    print ('slt_vel_jer: ')    
    aggregate(activation_slt_vel_jer_poor, activation_slt_vel_jer_ave, activation_slt_vel_jer_prof, three)

#pen down, duration and velocity
def pen_dur_vel():
    global three
    rule1_pen_dur_vel = np.fmax(in_penDown['short'], np.fmax(in_duration['long'], in_velocity['slow']))
    rule2_pen_dur_vel = np.fmax(in_penDown['average'], np.fmax(in_duration['average'], in_velocity['moderate']))
    rule3_pen_dur_vel = np.fmax(in_penDown['long'], np.fmax(in_duration['short'], in_velocity['fast']))
    #THEN
    activation_pen_dur_vel_poor = np.fmin(rule1_pen_dur_vel, score_hi)
    activation_pen_dur_vel_ave = np.fmin(rule2_pen_dur_vel, score_md)
    activation_pen_dur_vel_prof = np.fmin(rule3_pen_dur_vel, score_lo)
    
    print ('pen_dur_vel: ')    
    aggregate(activation_pen_dur_vel_poor, activation_pen_dur_vel_ave, activation_pen_dur_vel_prof, three)

#pen down, duration and jerk
def pen_dur_jer():
    global three
    rule1_pen_dur_jer = np.fmax(in_penDown['short'], np.fmax(in_duration['long'], in_jerk['high']))
    rule2_pen_dur_jer = np.fmax(in_penDown['average'], np.fmax(in_duration['average'], in_jerk['medium']))
    rule3_pen_dur_jer = np.fmax(in_penDown['long'], np.fmax(in_duration['short'], in_jerk['low']))
    #THEN
    activation_pen_dur_jer_poor = np.fmin(rule1_pen_dur_jer, score_hi)
    activation_pen_dur_jer_ave = np.fmin(rule2_pen_dur_jer, score_md)
    activation_pen_dur_jer_prof = np.fmin(rule3_pen_dur_jer, score_lo)
    
    print ('pen_dur_jer: ')    
    aggregate(activation_pen_dur_jer_poor, activation_pen_dur_jer_ave, activation_pen_dur_jer_prof, three)

#pen down, velocity and jerk
def pen_vel_jer():
    global three
    rule1_pen_vel_jer = np.fmax(in_penDown['short'], np.fmax(in_velocity['slow'], in_jerk['high']))
    rule2_pen_vel_jer = np.fmax(in_penDown['average'], np.fmax(in_velocity['moderate'], in_jerk['medium']))
    rule3_pen_vel_jer = np.fmax(in_penDown['long'], np.fmax(in_velocity['fast'], in_jerk['low']))
    #THEN
    activation_pen_vel_jer_poor = np.fmin(rule1_pen_vel_jer, score_hi)
    activation_pen_vel_jer_ave = np.fmin(rule2_pen_vel_jer, score_md)
    activation_pen_vel_jer_prof = np.fmin(rule3_pen_vel_jer, score_lo)
    
    print ('pen_vel_jer: ')    
    aggregate(activation_pen_vel_jer_poor, activation_pen_vel_jer_ave, activation_pen_vel_jer_prof, three)

#duration, velocity and jerk
def dur_vel_jer():
    global three
    rule1_dur_vel_jer = np.fmax(in_duration['long'], np.fmax(in_velocity['slow'], in_jerk['high']))
    rule2_dur_vel_jer = np.fmax(in_duration['average'], np.fmax(in_velocity['moderate'], in_jerk['medium']))
    rule3_dur_vel_jer = np.fmax(in_duration['short'], np.fmax(in_velocity['fast'], in_jerk['low']))
    #THEN
    activation_dur_vel_jer_poor = np.fmin(rule1_dur_vel_jer, score_hi)
    activation_dur_vel_jer_ave = np.fmin(rule2_dur_vel_jer, score_md)
    activation_dur_vel_jer_prof = np.fmin(rule3_dur_vel_jer, score_lo)
    
    print ('dur_vel_jer: ')    
    aggregate(activation_dur_vel_jer_poor, activation_dur_vel_jer_ave, activation_dur_vel_jer_prof, three)

######################################END######################################

#Rules Definition--4 variables
#####################################START#####################################
#IF pressure, slant, pen and duration
def pr_slt_pen_dur():
    global four
    rule1_pr_slt_pen_dur = np.fmax(np.fmax(in_pressure['hard'], in_slant['negative']), np.fmax(in_penDown['short'], in_duration['long']))
    rule2_pr_slt_pen_dur = np.fmax(in_pressure['moderate'], np.fmax(in_penDown['average'], in_duration['average']))
    rule3_pr_slt_pen_dur = np.fmax(np.fmax(in_pressure['light'], in_slant['vertical']), np.fmax(in_penDown['long'], in_duration['short']))
    #THEN
    activation_pr_slt_pen_dur_poor = np.fmin(rule1_pr_slt_pen_dur, score_hi)
    activation_pr_slt_pen_dur_ave = np.fmin(rule2_pr_slt_pen_dur, score_md)
    activation_pr_slt_pen_dur_prof = np.fmin(rule3_pr_slt_pen_dur, score_lo)
    
    print ('pr_slt_pen_dur: ')    
    aggregate(activation_pr_slt_pen_dur_poor, activation_pr_slt_pen_dur_ave, activation_pr_slt_pen_dur_prof, four)

#pressure, slant, pen and velocity
def pr_slt_pen_vel():
    global four
    rule1_pr_slt_pen_vel = np.fmax(np.fmax(in_pressure['hard'], in_slant['negative']), np.fmax(in_penDown['short'], in_velocity['slow']))
    rule2_pr_slt_pen_vel = np.fmax(in_pressure['moderate'], np.fmax(in_penDown['average'], in_velocity['moderate']))
    rule3_pr_slt_pen_vel = np.fmax(np.fmax(in_pressure['light'], in_slant['vertical']), np.fmax(in_penDown['long'], in_velocity['fast']))
    #THEN
    activation_pr_slt_pen_vel_poor = np.fmin(rule1_pr_slt_pen_vel, score_hi)
    activation_pr_slt_pen_vel_ave = np.fmin(rule2_pr_slt_pen_vel, score_md)
    activation_pr_slt_pen_vel_prof = np.fmin(rule3_pr_slt_pen_vel, score_lo)
    
    print ('pr_slt_pen_vel: ')    
    aggregate(activation_pr_slt_pen_vel_poor, activation_pr_slt_pen_vel_ave, activation_pr_slt_pen_vel_prof, four)

#pressure, slant, duration and velocity
def pr_slt_dur_vel():
    global four
    rule1_pr_slt_dur_vel = np.fmax(np.fmax(in_pressure['hard'], in_slant['negative']), np.fmax(in_duration['long'], in_velocity['slow']))
    rule2_pr_slt_dur_vel = np.fmax(in_pressure['moderate'], np.fmax(in_duration['average'], in_velocity['moderate']))
    rule3_pr_slt_dur_vel = np.fmax(np.fmax(in_pressure['light'], in_slant['vertical']), np.fmax(in_duration['short'], in_velocity['fast']))
    #THEN
    activation_pr_slt_dur_vel_poor = np.fmin(rule1_pr_slt_dur_vel, score_hi)
    activation_pr_slt_dur_vel_ave = np.fmin(rule2_pr_slt_dur_vel, score_md)
    activation_pr_slt_dur_vel_prof = np.fmin(rule3_pr_slt_dur_vel, score_lo)
    
    print ('pr_slt_dur_vel: ')    
    aggregate(activation_pr_slt_dur_vel_poor, activation_pr_slt_dur_vel_ave, activation_pr_slt_dur_vel_prof, four)

#pressure, slant, duration and velocity
def pr_slt_dur_jer():
    global four
    rule1_pr_slt_dur_jer = np.fmax(np.fmax(in_pressure['hard'], in_slant['negative']), np.fmax(in_duration['long'], in_jerk['high']))
    rule2_pr_slt_dur_jer = np.fmax(in_pressure['moderate'], np.fmax(in_duration['average'], in_jerk['medium']))
    rule3_pr_slt_dur_jer = np.fmax(np.fmax(in_pressure['light'], in_slant['vertical']), np.fmax(in_duration['short'], in_jerk['low']))
    #THEN
    activation_pr_slt_dur_jer_poor = np.fmin(rule1_pr_slt_dur_jer, score_hi)
    activation_pr_slt_dur_jer_ave = np.fmin(rule2_pr_slt_dur_jer, score_md)
    activation_pr_slt_dur_jer_prof = np.fmin(rule3_pr_slt_dur_jer, score_lo)
    
    print ('pr_slt_dur_jer: ')    
    aggregate(activation_pr_slt_dur_jer_poor, activation_pr_slt_dur_jer_ave, activation_pr_slt_dur_jer_prof, four)
    
#pressure, slant, pen and jerk
def pr_slt_pen_jer():
    global four
    rule1_pr_slt_pen_jer = np.fmax(np.fmax(in_pressure['hard'], in_slant['negative']), np.fmax(in_penDown['short'], in_jerk['high']))
    rule2_pr_slt_pen_jer = np.fmax(in_pressure['moderate'], np.fmax(in_penDown['average'], in_jerk['medium']))
    rule3_pr_slt_pen_jer = np.fmax(np.fmax(in_pressure['light'], in_slant['vertical']), np.fmax(in_penDown['long'], in_jerk['low']))
    #THEN
    activation_pr_slt_pen_jer_poor = np.fmin(rule1_pr_slt_pen_jer, score_hi)
    activation_pr_slt_pen_jer_ave = np.fmin(rule2_pr_slt_pen_jer, score_md)
    activation_pr_slt_pen_jer_prof = np.fmin(rule3_pr_slt_pen_jer, score_lo)
    
    print ('pr_slt_pen_jer: ')    
    aggregate(activation_pr_slt_pen_jer_poor, activation_pr_slt_pen_jer_ave, activation_pr_slt_pen_jer_prof, four)

#pressure, pen down, duration and velocity
def pr_pen_dur_vel():
    global four
    rule1_pr_pen_dur_vel = np.fmax(np.fmax(in_pressure['hard'], in_penDown['short']), np.fmax(in_duration['long'], in_velocity['slow']))
    rule2_pr_pen_dur_vel = np.fmax(np.fmax(in_pressure['moderate'], in_penDown['average']), np.fmax(in_duration['average'], in_velocity['moderate']))
    rule3_pr_pen_dur_vel = np.fmax(np.fmax(in_pressure['light'], in_penDown['long']), np.fmax(in_duration['short'], in_velocity['fast']))
    #THEN
    activation_pr_pen_dur_vel_poor = np.fmin(rule1_pr_pen_dur_vel, score_hi)
    activation_pr_pen_dur_vel_ave = np.fmin(rule2_pr_pen_dur_vel, score_md)
    activation_pr_pen_dur_vel_prof = np.fmin(rule3_pr_pen_dur_vel, score_lo)
    
    print ('pr_pen_dur_vel: ')    
    aggregate(activation_pr_pen_dur_vel_poor, activation_pr_pen_dur_vel_ave, activation_pr_pen_dur_vel_prof, four)

#pressure, pen down, duration and jerk
def pr_pen_dur_jer():
    global four
    rule1_pr_pen_dur_jer = np.fmax(np.fmax(in_pressure['hard'], in_penDown['short']), np.fmax(in_duration['long'], in_jerk['high']))
    rule2_pr_pen_dur_jer = np.fmax(np.fmax(in_pressure['moderate'], in_penDown['average']), np.fmax(in_duration['average'], in_jerk['medium']))
    rule3_pr_pen_dur_jer = np.fmax(np.fmax(in_pressure['light'], in_penDown['long']), np.fmax(in_duration['short'], in_jerk['low']))
    #THEN
    activation_pr_pen_dur_jer_poor = np.fmin(rule1_pr_pen_dur_jer, score_hi)
    activation_pr_pen_dur_jer_ave = np.fmin(rule2_pr_pen_dur_jer, score_md)
    activation_pr_pen_dur_jer_prof = np.fmin(rule3_pr_pen_dur_jer, score_lo)
    
    print ('pr_pen_dur_jer: ')    
    aggregate(activation_pr_pen_dur_jer_poor, activation_pr_pen_dur_jer_ave, activation_pr_pen_dur_jer_prof, four)

#pressure, duration, velocity and jerk
def pr_dur_vel_jer():
    global four
    rule1_pr_dur_vel_jer = np.fmax(np.fmax(in_pressure['hard'], in_duration['long']), np.fmax(in_velocity['slow'], in_jerk['high']))
    rule2_pr_dur_vel_jer = np.fmax(np.fmax(in_pressure['moderate'], in_duration['average']), np.fmax(in_velocity['moderate'], in_jerk['medium']))
    rule3_pr_dur_vel_jer = np.fmax(np.fmax(in_pressure['light'], in_duration['short']), np.fmax(in_velocity['fast'], in_jerk['low']))
    #THEN
    activation_pr_dur_vel_jer_poor = np.fmin(rule1_pr_dur_vel_jer, score_hi)
    activation_pr_dur_vel_jer_ave = np.fmin(rule2_pr_dur_vel_jer, score_md)
    activation_pr_dur_vel_jer_prof = np.fmin(rule3_pr_dur_vel_jer, score_lo)

    print ('pr_dur_vel_jer: ')    
    aggregate(activation_pr_dur_vel_jer_poor, activation_pr_dur_vel_jer_ave, activation_pr_dur_vel_jer_prof, four)

#slant, pen down, duration and velocity
def slt_pen_dur_vel():
    global four
    rule1_slt_pen_dur_vel = np.fmax(np.fmax(in_slant['negative'], in_penDown['short']), np.fmax(in_duration['long'], in_velocity['slow']))
    rule2_slt_pen_dur_vel = np.fmax(in_penDown['average'], np.fmax(in_duration['average'], in_velocity['moderate']))
    rule3_slt_pen_dur_vel = np.fmax(np.fmax(in_slant['vertical'], in_penDown['long']), np.fmax(in_duration['short'], in_velocity['fast']))
    #THEN
    activation_slt_pen_dur_vel_poor = np.fmin(rule1_slt_pen_dur_vel, score_hi)
    activation_slt_pen_dur_vel_ave = np.fmin(rule2_slt_pen_dur_vel, score_md)
    activation_slt_pen_dur_vel_prof = np.fmin(rule3_slt_pen_dur_vel, score_lo)
    
    print ('slt_pen_dur_vel: ')    
    aggregate(activation_slt_pen_dur_vel_poor, activation_slt_pen_dur_vel_ave, activation_slt_pen_dur_vel_prof, four)

#slant, pen down, duration and jerk
def slt_pen_dur_jer():
    global four
    rule1_slt_pen_dur_jer = np.fmax(np.fmax(in_slant['negative'], in_penDown['short']), np.fmax(in_duration['long'], in_jerk['high']))
    rule2_slt_pen_dur_jer = np.fmax(in_penDown['average'], np.fmax(in_duration['average'], in_jerk['medium']))
    rule3_slt_pen_dur_jer = np.fmax(np.fmax(in_slant['vertical'], in_penDown['long']), np.fmax(in_duration['short'], in_jerk['low']))
    #THEN
    activation_slt_pen_dur_jer_poor = np.fmin(rule1_slt_pen_dur_jer, score_hi)
    activation_slt_pen_dur_jer_ave = np.fmin(rule2_slt_pen_dur_jer, score_md)
    activation_slt_pen_dur_jer_prof = np.fmin(rule3_slt_pen_dur_jer, score_lo)
    
    print ('slt_pen_dur_jer: ')    
    aggregate(activation_slt_pen_dur_jer_poor, activation_slt_pen_dur_jer_ave, activation_slt_pen_dur_jer_prof, four)

#slant, duration, velocity and jerk
def slt_dur_vel_jer():
    global four
    rule1_slt_dur_vel_jer = np.fmax(np.fmax(in_slant['negative'], in_duration['long']), np.fmax(in_velocity['slow'], in_jerk['high']))
    rule2_slt_dur_vel_jer = np.fmax(in_duration['average'], np.fmax(in_velocity['moderate'], in_jerk['medium']))
    rule3_slt_dur_vel_jer = np.fmax(np.fmax(in_slant['vertical'], in_duration['short']), np.fmax(in_velocity['fast'], in_jerk['low']))
    #THEN
    activation_slt_dur_vel_jer_poor = np.fmin(rule1_slt_dur_vel_jer, score_hi)
    activation_slt_dur_vel_jer_ave = np.fmin(rule2_slt_dur_vel_jer, score_md)
    activation_slt_dur_vel_jer_prof = np.fmin(rule3_slt_dur_vel_jer, score_lo)
    
    print ('slt_dur_vel_jer: ')    
    aggregate(activation_slt_dur_vel_jer_poor, activation_slt_dur_vel_jer_ave, activation_slt_dur_vel_jer_prof, four)


#pen down, duration, velocity and jerk
def pen_dur_vel_jer():
    global four
    rule1_pen_dur_vel_jer = np.fmax(np.fmax(in_penDown['short'], in_duration['long']), np.fmax(in_velocity['slow'], in_jerk['high']))
    rule2_pen_dur_vel_jer = np.fmax(np.fmax(in_penDown['average'], in_duration['average']), np.fmax(in_velocity['moderate'], in_jerk['medium']))
    rule3_pen_dur_vel_jer = np.fmax(np.fmax(in_penDown['long'], in_duration['short']), np.fmax(in_velocity['fast'], in_jerk['low']))
    #THEN
    activation_pen_dur_vel_jer_poor = np.fmin(rule1_pen_dur_vel_jer, score_hi)
    activation_pen_dur_vel_jer_ave = np.fmin(rule2_pen_dur_vel_jer, score_md)
    activation_pen_dur_vel_jer_prof = np.fmin(rule3_pen_dur_vel_jer, score_lo)
    
    print ('pen_dur_vel_jer: ')    
    aggregate(activation_pen_dur_vel_jer_poor, activation_pen_dur_vel_jer_ave, activation_pen_dur_vel_jer_prof, four)

######################################END######################################

#Rules Definition--5 variables
#####################################START#####################################
#IF pressure, slant, pen, duration and velocity
def pr_slt_pen_dur_vel():
    global five
    rule1_pr_slt_pen_dur_vel = np.fmax(in_pressure['hard'], np.fmax(np.fmax(in_slant['negative'], in_penDown['short']), np.fmax(in_duration['long'], in_velocity['slow'])))
    rule2_pr_slt_pen_dur_vel = np.fmax(np.fmax(in_pressure['moderate'], in_penDown['average']), np.fmax(in_duration['average'], in_velocity['moderate']))
    rule3_pr_slt_pen_dur_vel = np.fmax(in_pressure['light'], np.fmax(np.fmax(in_slant['vertical'], in_penDown['long']), np.fmax(in_duration['short'], in_velocity['fast'])))
    #THEN
    activation_pr_slt_pen_dur_vel_poor = np.fmin(rule1_pr_slt_pen_dur_vel, score_hi)
    activation_pr_slt_pen_dur_vel_ave = np.fmin(rule2_pr_slt_pen_dur_vel, score_md)
    activation_pr_slt_pen_dur_vel_prof = np.fmin(rule3_pr_slt_pen_dur_vel, score_lo)
    
    print ('pr_slt_pen_dur_vel: ')    
    aggregate(activation_pr_slt_pen_dur_vel_poor, activation_pr_slt_pen_dur_vel_ave, activation_pr_slt_pen_dur_vel_prof, five)

#pressure, slant, pen, duration and jerk
def pr_slt_pen_dur_jer():
    global five
    rule1_pr_slt_pen_dur_jer = np.fmax(in_pressure['hard'], np.fmax(np.fmax(in_slant['negative'], in_penDown['short']), np.fmax(in_duration['long'], in_jerk['high'])))
    rule2_pr_slt_pen_dur_jer = np.fmax(np.fmax(in_pressure['moderate'], in_penDown['average']), np.fmax(in_duration['average'], in_jerk['medium']))
    rule3_pr_slt_pen_dur_jer = np.fmax(in_pressure['light'], np.fmax(np.fmax(in_slant['vertical'], in_penDown['long']), np.fmax(in_duration['short'], in_jerk['low'])))
    #THEN
    activation_pr_slt_pen_dur_jer_poor = np.fmin(rule1_pr_slt_pen_dur_jer, score_hi)
    activation_pr_slt_pen_dur_jer_ave = np.fmin(rule2_pr_slt_pen_dur_jer, score_md)
    activation_pr_slt_pen_dur_jer_prof = np.fmin(rule3_pr_slt_pen_dur_jer, score_lo)

    print ('pr_slt_pen_dur_jer: ')    
    aggregate(activation_pr_slt_pen_dur_jer_poor, activation_pr_slt_pen_dur_jer_ave, activation_pr_slt_pen_dur_jer_prof, five)

#slant, pen down, duration, velocity and jerk
def slt_pen_dur_vel_jer():
    global five
    rule1_slt_pen_dur_vel_jer = np.fmax(in_slant['negative'], np.fmax(np.fmax(in_penDown['short'], in_duration['long']), np.fmax(in_velocity['slow'], in_jerk['high'])))
    rule2_slt_pen_dur_vel_jer = np.fmax(np.fmax(in_penDown['average'], in_duration['average']), np.fmax(in_velocity['moderate'], in_jerk['medium']))
    rule3_slt_pen_dur_vel_jer = np.fmax(in_slant['vertical'], np.fmax(np.fmax(in_penDown['long'], in_duration['short']), np.fmax(in_velocity['fast'], in_jerk['low'])))
    #THEN
    activation_slt_pen_dur_vel_jer_poor = np.fmin(rule1_slt_pen_dur_vel_jer, score_hi)
    activation_slt_pen_dur_vel_jer_ave = np.fmin(rule2_slt_pen_dur_vel_jer, score_md)
    activation_slt_pen_dur_vel_jer_prof = np.fmin(rule3_slt_pen_dur_vel_jer, score_lo)
    
    print ('slt_pen_dur_vel_jer: ')    
    aggregate(activation_slt_pen_dur_vel_jer_poor, activation_slt_pen_dur_vel_jer_ave, activation_slt_pen_dur_vel_jer_prof, five)

#pen down, duration, velocity, jerk and pressure
def pen_dur_vel_jer_pr():
    global five
    rule1_pen_dur_vel_jer_pr = np.fmax(in_penDown['short'], np.fmax(np.fmax(in_duration['long'], in_velocity['slow']), np.fmax(in_jerk['high'], in_pressure['hard'])))
    rule2_pen_dur_vel_jer_pr = np.fmax(in_penDown['average'], np.fmax(np.fmax(in_duration['average'], in_velocity['moderate']), np.fmax(in_jerk['medium'], in_pressure['moderate'])))
    rule3_pen_dur_vel_jer_pr = np.fmax(in_penDown['long'], np.fmax(np.fmax(in_duration['short'], in_velocity['fast']), np.fmax(in_jerk['low'], in_pressure['light'])))
    #THEN
    activation_pen_dur_vel_jer_pr_poor = np.fmin(rule1_pen_dur_vel_jer_pr, score_hi)
    activation_pen_dur_vel_jer_pr_ave = np.fmin(rule2_pen_dur_vel_jer_pr, score_md)
    activation_pen_dur_vel_jer_pr_prof = np.fmin(rule3_pen_dur_vel_jer_pr, score_lo)
    
    print ('pen_dur_vel_jer_pr: ')    
    aggregate(activation_pen_dur_vel_jer_pr_poor, activation_pen_dur_vel_jer_pr_ave, activation_pen_dur_vel_jer_pr_prof, five)

#slant, duration, velocity, jerk and pressure
def slt_dur_vel_jer_pr():
    global five
    rule1_slt_dur_vel_jer_pr = np.fmax(in_slant['negative'], np.fmax(np.fmax(in_duration['long'], in_velocity['slow']), np.fmax(in_jerk['high'], in_pressure['hard'])))
    rule2_slt_dur_vel_jer_pr = np.fmax(np.fmax(in_duration['average'], in_velocity['moderate']), np.fmax(in_jerk['medium'], in_pressure['moderate']))
    rule3_slt_dur_vel_jer_pr = np.fmax(in_slant['vertical'], np.fmax(np.fmax(in_duration['short'], in_velocity['fast']), np.fmax(in_jerk['low'], in_pressure['light'])))
    #THEN
    activation_slt_dur_vel_jer_pr_poor = np.fmin(rule1_slt_dur_vel_jer_pr, score_hi)
    activation_slt_dur_vel_jer_pr_ave = np.fmin(rule2_slt_dur_vel_jer_pr, score_md)
    activation_slt_dur_vel_jer_pr_prof = np.fmin(rule3_slt_dur_vel_jer_pr, score_lo)
    
    print ('slt_dur_vel_jer_pr: ')    
    aggregate(activation_slt_dur_vel_jer_pr_poor, activation_slt_dur_vel_jer_pr_ave, activation_slt_dur_vel_jer_pr_prof, five)

#pen down, slant, velocity, jerk and pressure
def pen_slt_vel_jer_pr():
    global five
    rule1_pen_slt_vel_jer_pr = np.fmax(in_penDown['short'], np.fmax(np.fmax(in_slant['negative'], in_velocity['slow']), np.fmax(in_jerk['high'], in_pressure['hard'])))
    rule2_pen_slt_vel_jer_pr = np.fmax(np.fmax(in_penDown['average'], in_velocity['moderate']), np.fmax(in_jerk['medium'], in_pressure['moderate']))
    rule3_pen_slt_vel_jer_pr = np.fmax(in_penDown['long'], np.fmax(np.fmax(in_slant['vertical'], in_velocity['fast']), np.fmax(in_jerk['low'], in_pressure['light'])))
    #THEN
    activation_pen_slt_vel_jer_pr_poor = np.fmin(rule1_pen_slt_vel_jer_pr, score_hi)
    activation_pen_slt_vel_jer_pr_ave = np.fmin(rule2_pen_slt_vel_jer_pr, score_md)
    activation_pen_slt_vel_jer_pr_prof = np.fmin(rule3_pen_slt_vel_jer_pr, score_lo)
    
    print ('pen_slt_vel_jer_pr: ')    
    aggregate(activation_pen_slt_vel_jer_pr_poor, activation_pen_slt_vel_jer_pr_ave, activation_pen_slt_vel_jer_pr_prof, five)

#pen down, slant, duration, jerk and pressure
def pen_slt_dur_jer_pr():
    global five
    rule1_pen_slt_dur_jer_pr = np.fmax(in_penDown['short'], np.fmax(np.fmax(in_slant['negative'], in_duration['long']), np.fmax(in_jerk['high'], in_pressure['hard'])))
    rule2_pen_slt_dur_jer_pr = np.fmax(np.fmax(in_penDown['average'], in_duration['average']), np.fmax(in_jerk['medium'], in_pressure['moderate']))
    rule3_pen_slt_dur_jer_pr = np.fmax(in_penDown['long'], np.fmax(np.fmax(in_slant['vertical'], in_duration['short']), np.fmax(in_jerk['low'], in_pressure['light'])))
    #THEN
    activation_pen_slt_dur_jer_pr_poor = np.fmin(rule1_pen_slt_dur_jer_pr, score_hi)
    activation_pen_slt_dur_jer_pr_ave = np.fmin(rule2_pen_slt_dur_jer_pr, score_md)
    activation_pen_slt_dur_jer_pr_prof = np.fmin(rule3_pen_slt_dur_jer_pr, score_lo)
    
    print ('pen_slt_dur_jer_pr: ')    
    aggregate(activation_pen_slt_dur_jer_pr_poor, activation_pen_slt_dur_jer_pr_ave, activation_pen_slt_dur_jer_pr_prof, five)

######################################END######################################

#Rules Definition--6 variables
#####################################START#####################################
#IF pressure, slant, pen, duration, velocity and jerk
def pr_slt_pen_dur_vel_jer():
    global six
    rule1_pr_slt_pen_dur_vel_jer = np.fmax(np.fmax(in_pressure['hard'], in_slant['negative']), np.fmax(np.fmax(in_penDown['short'], in_duration['long']), np.fmax(in_velocity['slow'], in_jerk['high'])))
    rule2_pr_slt_pen_dur_vel_jer = np.fmax(in_pressure['moderate'], np.fmax(np.fmax(in_penDown['average'], in_duration['average']), np.fmax(in_velocity['moderate'], in_jerk['medium'])))
    rule3_pr_slt_pen_dur_vel_jer = np.fmax(np.fmax(in_pressure['light'], in_slant['vertical']), np.fmax(np.fmax(in_penDown['long'], in_duration['short']), np.fmax(in_velocity['fast'], in_jerk['low'])))
    #THEN
    activation_pr_slt_pen_dur_vel_jer_poor = np.fmin(rule1_pr_slt_pen_dur_vel_jer, score_hi)
    activation_pr_slt_pen_dur_vel_jer_ave = np.fmin(rule2_pr_slt_pen_dur_vel_jer, score_md)
    activation_pr_slt_pen_dur_vel_jer_prof = np.fmin(rule3_pr_slt_pen_dur_vel_jer, score_lo)
    
    print ('pr_slt_pen_dur_vel_jer: ')    
    aggregate(activation_pr_slt_pen_dur_vel_jer_poor, activation_pr_slt_pen_dur_vel_jer_ave, activation_pr_slt_pen_dur_vel_jer_prof, six)
    
######################################END######################################

    
#################################MAIN CODE#####################################
#open CSV file    
filename = askopenfilename() 
df = pd.read_csv(filename)
print (filename)
#reading each column in the file
#both TRIALS
column_trial = df['Trial']
column_duration = df['Duration']
column_slant = df['Slant']
column_penDown = df['RelativePenDownDuration']
column_velocity = df['AverageAbsoluteVelocity']
column_jerk = df['AverageNormalizedJerkPerTrial']
column_pressure = df['AveragePenPressure']

#fuzzification
global pressure, slant, penDown, duration, velocity, jerk, score
global pressure_lo, pressure_md, pressure_hi, slant_lo, slant_md, slant_hi, \
penDown_lo, penDown_md, penDown_hi, duration_lo, duration_md, duration_hi, \
velocity_lo, velocity_md, velocity_hi, jerk_lo, jerk_md, jerk_hi, \
score_lo, score_md, score_hi
one = []
two = []
three = []
four = []
five = []
six =[]
    
#antecedent(input) measured per stroke
pressure = np.arange(0, 1023.001,0.001)
slant = np.arange(-3.15, 3.151, 0.001)
penDown = np.arange(0, 1.001, 0.001)
duration = np.arange(0, 10.001, 0.001)
velocity = np.arange(0, 22.001, 0.001)
jerk = np.arange(0, 7000.001,  0.001)
#consequent (output)
score = np.arange(0, 31, 1)
    
#membership functions
pressure_lo = fz.trimf(pressure, [0, 0, 511.000])
pressure_md = fz.trimf(pressure, [0, 511.000, 1023])
pressure_hi = fz.trimf(pressure, [511.000, 1023, 1023])
slant_lo = fz.trimf(slant, [-3.15, -3.15, 0])
slant_md = fz.trimf(slant, [-3.15, 0, 3.15])
slant_hi = fz.trimf(slant, [0, 3.15, 3.15])
penDown_lo = fz.trimf(penDown, [0, 0, 0.500])
penDown_md = fz.trimf(penDown, [0, 0.500, 1])
penDown_hi = fz.trimf(penDown, [0.500, 1, 1])
duration_lo = fz.trimf(duration, [0, 0, 5.000])
duration_md = fz.trimf(duration, [0, 5.000, 10])
duration_hi = fz.trimf(duration, [5.000, 10, 10])
velocity_lo = fz.trimf(velocity, [0, 0, 11])
velocity_md = fz.trimf(velocity, [0, 11, 22])
velocity_hi = fz.trimf(velocity, [11, 22, 22])
jerk_lo = fz.trimf(jerk, [0, 0, 3500])
jerk_md = fz.trimf(jerk, [0, 3500, 7000])
jerk_hi = fz.trimf(jerk, [3500, 7000, 7000])
#membership (output)
score_lo = fz.trimf(score, [0, 0, 15])
score_md = fz.trimf(score, [0, 15, 30])
score_hi = fz.trimf(score, [15, 30, 30])

#plotting of fuzzified data
fig, (pl_pr, pl_slt, pl_pen, pl_dur, pl_vel, pl_jer, pl_sco) = plt.subplots(nrows=7, figsize=(8,25))

pl_pr.plot(pressure, pressure_lo, 'b', linewidth=1.5, label='light')
pl_pr.plot(pressure, pressure_md, 'g', linewidth=1.5, label='moderate')
pl_pr.plot(pressure, pressure_hi, 'r', linewidth=1.5, label='hard')
pl_pr.set_title('Pressure')
pl_pr.legend()

pl_slt.plot(slant, slant_lo, 'b', linewidth=1.5, label='negDiagonal')
pl_slt.plot(slant, slant_md, 'g', linewidth=1.5, label='vertical')
pl_slt.plot(slant, slant_hi, 'r', linewidth=1.5, label='posDiagonal')
pl_slt.set_title('Slant')
pl_slt.legend()

pl_pen.plot(penDown, penDown_lo, 'b', linewidth=1.5, label='short')
pl_pen.plot(penDown, penDown_md, 'g', linewidth=1.5, label='average')
pl_pen.plot(penDown, penDown_hi, 'r', linewidth=1.5, label='long')
pl_pen.set_title('PenDown')
pl_pen.legend()

pl_dur.plot(duration, duration_lo, 'b', linewidth=1.5, label='short')
pl_dur.plot(duration, duration_md, 'g', linewidth=1.5, label='average')
pl_dur.plot(duration, duration_hi, 'r', linewidth=1.5, label='long')
pl_dur.set_title('Duration')
pl_dur.legend()

pl_vel.plot(velocity, velocity_lo, 'b', linewidth=1.5, label='slow')
pl_vel.plot(velocity, velocity_md, 'g', linewidth=1.5, label='moderate')
pl_vel.plot(velocity, velocity_hi, 'r', linewidth=1.5, label='fast')
pl_vel.set_title('Velocity')
pl_vel.legend()

pl_jer.plot(jerk, jerk_lo, 'b', linewidth=1.5, label='low')
pl_jer.plot(jerk, jerk_md, 'g', linewidth=1.5, label='medium')
pl_jer.plot(jerk, jerk_hi, 'r', linewidth=1.5, label='high')
pl_jer.set_title('Jerk')
pl_jer.legend()

pl_sco.plot(score, score_lo, 'b', linewidth=1.5, label='low')
pl_sco.plot(score, score_md, 'g', linewidth=1.5, label='medium')
pl_sco.plot(score, score_hi, 'r', linewidth=1.5, label='high')
pl_sco.set_title('Evaluation Score')
pl_sco.legend()

for ln in (pl_pr, pl_slt, pl_pen, pl_dur, pl_vel, pl_jer, pl_sco):
    ln.spines['top'].set_visible(False)
    ln.spines['right'].set_visible(False)
    ln.get_xaxis().tick_bottom()
    ln.get_yaxis().tick_left()

plt.tight_layout()

#definition of values
#BOTH trials
in_duration = feature_duration(getMean(column_duration))
in_slant = feature_slant(getMean(column_slant))
in_penDown = feature_penDown(getMean(column_penDown))
in_velocity = feature_velocity(getMean(column_velocity))
in_jerk = feature_jerk(getMean(column_jerk))
in_pressure = feature_pressure(getMean(column_pressure))

#1 variable--rules output (6 rules)
print("--------1 variable--------")
pr()
slt()
pen()
dur()
vel()
jer()

#2 variables--rules output (15 rules)
print("--------2 variables--------")
pr_slt()
pr_pen()
pr_dur()
pr_vel()
pr_jer()
slt_pen()
slt_dur()
slt_vel()
slt_jer()
pen_dur()
pen_vel()
pen_jer()
dur_vel()
dur_jer()
vel_jer()

#3 variables--rules output (20 rules)
print("--------3 variables--------")
pr_slt_pen()
pr_slt_dur()
pr_slt_vel()
pr_slt_jer()
pr_pen_dur()
pr_pen_vel()
pr_pen_jer()
pr_dur_vel()
pr_dur_jer()
pr_vel_jer()
slt_pen_dur()
slt_pen_vel()
slt_pen_jer()
slt_dur_vel()
slt_dur_jer()
slt_vel_jer()
pen_dur_vel()
pen_dur_jer()
pen_vel_jer()
dur_vel_jer()

#4 variables--rules output (10 rules)
print("--------4 variables--------")
pr_slt_pen_dur()
pr_slt_pen_vel()
pr_slt_pen_jer()
pr_slt_dur_vel()
pr_slt_dur_jer()
pr_pen_dur_vel()
pr_pen_dur_jer()
pr_dur_vel_jer()
slt_pen_dur_vel()
slt_pen_dur_jer()
slt_dur_vel_jer()
pen_dur_vel_jer()

#5 variables--rules output (7 rules)
print("--------5 variables--------")
pr_slt_pen_dur_vel()
pr_slt_pen_dur_jer()
slt_pen_dur_vel_jer()
pen_dur_vel_jer_pr()
slt_dur_vel_jer_pr()
pen_slt_vel_jer_pr()
pen_slt_dur_jer_pr()

#6 variable--rules output (1 rule)
print("--------6 variables--------")
pr_slt_pen_dur_vel_jer()

createCSV(one, two, three, four, five, six)