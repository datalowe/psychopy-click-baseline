#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.5),
    on Mon Nov  9 14:40:16 2020
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

### SET EXPERIMENT CONSTANTS ###
# width/height of area where targets
# can appear, in degrees
AREA_WIDTH_DEG = 14
AREA_HEIGHT_DEG = 14

# target text (digit) size, in degrees
TARGET_SIZE_DEG = 1.5
# target text neutral (non-clicked) color
TARGET_NEUTRAL_COL = "#FFFFFF"
# target text clicked color
TARGET_CLICKED_COL = "#BBFFBB"

# large/medium/small instructions text size,
# in degrees
TXT_SIZE_L = 2
TXT_SIZE_M = 0.8
TXT_SIZE_S = 0.6

# number of trials to run
NUM_TRIALS = 3

# number of targets (digits/numbers) to use, ie how many
# targets to show per trial. target numbering always
# starts from 1.
NUM_TARGETS = 9

### END SET EXPERIMENT CONSTANTS ###


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2020.2.5'
expName = 'click_baseline'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sort_keys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/workingman/Documents/ASD_and_Synesthesia/Python/psychopy/PsychoPy projects/click_baseline/click_baseline.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.DEBUG)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(
    size=[1280, 800], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='macMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "instructions"
instructionsClock = core.Clock()
text_instructions = visual.TextStim(win=win, name='text_instructions',
    text='Din uppgift i det här testet är att så snabbt du kan klicka med musen på siffror som kommer upp på skärmen nummordning. Klicka först på 1, sen 2, sen fortsätt så upp till den högsta siffran.\n\nAnvänd vänster musknapp.\n\nTestet tar ca 3 minuter att genomföra.',
    font='Arial',
    units='deg', pos=(0, 0), height=TXT_SIZE_M, wrapWidth=25, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
mouse_instructions = event.Mouse(win=win)
x, y = [None, None]
mouse_instructions.mouseClock = core.Clock()
text_go = visual.TextStim(win=win, name='text_go',
    text='Klicka här när du läst klart för att börja testet',
    font='Arial',
    units='deg', pos=(0, -6), height=TXT_SIZE_S, wrapWidth=25, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "trial"
trialClock = core.Clock()
class Point:
    """
    Represents (x, y) coordinates in 2-dimensional space.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __sub__(self, other):
        """
        Returns the euclidean distance between this and the
        other passed point.
        """
        dist = ((self.x - other.x)**2 + (self.y - other.y)**2)**(1/2)
        return dist
    
    def __str__(self):
        """
        Returns string representation '(x, y)' of this point.
        """
        return "({}, {})".format(self.x, self.y)
    
    def __repr__(self):
        """
        Returns description of how this point can be
        recreated.
        """
        return "Point({}, {})".format(self.x, self.y)
    
    def as_tuple(self):
        """
        Returns a (x, y) tuple with this point's x/y coordinates.
        """
        return self.x, self.y

def points_collide(point_ls, new_point):
    """
    Goes through a list of Point instances and compares
    the new_point to see if the new_point is too close
    to one of the points in the point_ls. Returns True if
    there is a collision, otherwise False.
    """
    collides = False
    for old_point in point_ls:
        if (new_point - old_point) < TARGET_SIZE_DEG:
            collides = True
            break
    return collides

def gen_rand_point():
    """
    Generates a random point within the target area.
    """
    x_coord = randint(-AREA_WIDTH_DEG//2, AREA_WIDTH_DEG//2)
    y_coord = randint(-AREA_HEIGHT_DEG//2, AREA_HEIGHT_DEG//2)
    return Point(x_coord, y_coord)

def gen_point_ls():
    """
    Generate a list of (x, y) coordinate Point instances until
    there are as many instances as there should be targets/trial.
    """
    point_ls = []
    while len(point_ls) < NUM_TARGETS:
        new_point = gen_rand_point()
        if not points_collide(point_ls, new_point):
            point_ls.append(new_point)
    return point_ls

def gen_trial_ls():
    """
    Generates a list of lists, where each inner list corresponds to
    points describing the placements of all targets in a trial.
    """
    trial_ls = []
    while len(trial_ls) < NUM_TRIALS:
        point_ls = gen_point_ls()
        trial_ls.append(point_ls)
    return trial_ls

trial_ls = gen_trial_ls()

# generate one TextStim instance for each target
# (numbering starting from 1)
targets = []
for i in range(1, NUM_TARGETS+1):
    new_target = visual.TextStim(
        win=win, 
        name='text_target_{}'.format(i),
        text='{}'.format(i),
        font='Arial',
        units='deg', 
        pos=(0, 0), 
        height=TARGET_SIZE_DEG, 
        wrapWidth=None, ori=0, 
        color=TARGET_NEUTRAL_COL, 
        colorSpace='rgb', 
        opacity=1, 
        languageStyle='LTR',
        depth=-1.0)
    targets.append(new_target)

# initialize trial counter
trial_counter = 0
mouse_trial = event.Mouse(win=win)
x, y = [None, None]
mouse_trial.mouseClock = core.Clock()

# Initialize components for Routine "end_routine"
end_routineClock = core.Clock()
text_end = visual.TextStim(win=win, name='text_end',
    text='Nu är du klar. Tack!',
    font='Arial',
    units='deg', pos=(0, 0), height=TXT_SIZE_L, wrapWidth=25, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "instructions"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_instructions
mouse_instructions.clicked_name = []
gotValidClick = False  # until a click is received
# keep track of which components have finished
instructionsComponents = [text_instructions, mouse_instructions, text_go]
for thisComponent in instructionsComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instructionsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instructions"-------
while continueRoutine:
    # get current time
    t = instructionsClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instructionsClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_instructions* updates
    if text_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_instructions.frameNStart = frameN  # exact frame index
        text_instructions.tStart = t  # local t and not account for scr refresh
        text_instructions.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_instructions, 'tStartRefresh')  # time at next scr refresh
        text_instructions.setAutoDraw(True)
    # *mouse_instructions* updates
    if mouse_instructions.status == NOT_STARTED and t >= 0.5-frameTolerance:
        # keep track of start time/frame for later
        mouse_instructions.frameNStart = frameN  # exact frame index
        mouse_instructions.tStart = t  # local t and not account for scr refresh
        mouse_instructions.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_instructions, 'tStartRefresh')  # time at next scr refresh
        mouse_instructions.status = STARTED
        mouse_instructions.mouseClock.reset()
        prevButtonState = mouse_instructions.getPressed()  # if button is down already this ISN'T a new click
    if mouse_instructions.status == STARTED:  # only update if started and not finished!
        buttons = mouse_instructions.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                # check if the mouse was inside our 'clickable' objects
                gotValidClick = False
                for obj in [text_go]:
                    if obj.contains(mouse_instructions):
                        gotValidClick = True
                        mouse_instructions.clicked_name.append(obj.name)
                if gotValidClick:  # abort routine on response
                    continueRoutine = False
    
    # *text_go* updates
    if text_go.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_go.frameNStart = frameN  # exact frame index
        text_go.tStart = t  # local t and not account for scr refresh
        text_go.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_go, 'tStartRefresh')  # time at next scr refresh
        text_go.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instructions"-------
for thisComponent in instructionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# store data for thisExp (ExperimentHandler)
thisExp.nextEntry()
# the Routine "instructions" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=NUM_TRIALS, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec('{} = thisTrial[paramName]'.format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "trial"-------
    continueRoutine = True
    # update component parameters for each repeat
    # fetch the list of Point instances representing the
    # (x, y) coordinates that are to be used for targets
    # during this trial
    point_ls = trial_ls[trial_counter]
    
    # assign positions to the targets and reset their
    # colors to white
    for i, target in enumerate(targets):
        target.pos = point_ls[i].as_tuple()
        target.color = TARGET_NEUTRAL_COL
    
    # make a shallow copy of the targets list that keeps
    # track of which targets haven't been clicked on yet
    nonclicked_targets = targets[:]
    
    # set first target to be clicked
    click_target = nonclicked_targets.pop(0)
    
    # reset list of times when clicks occur
    response_times = []
    
    # fetch the routine start time
    trial_start_time = globalClock.getTime()
    
    # setup some python lists for storing info about the mouse_trial
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    trialComponents = [mouse_trial]
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "trial"-------
    while continueRoutine:
        # get current time
        t = trialClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trialClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # draw all of the targets
        for target in targets:
            target.draw()
        
        # check if mouse button has been clicked, and if so,
        # if any of the targets were clicked
        buttons = mouse_trial.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                # check if the mouse was inside of target
                if click_target.contains(mouse_trial):
                    # change color of target to indicate it's been clicked
                    click_target.color = TARGET_CLICKED_COL
                    # save the click time
                    response_times.append(trialClock.getTime())
                    # if there are targets left to click, make the next target in line clickable
                    # otherwise, end the routine
                    if len(nonclicked_targets) > 0:
                        click_target = nonclicked_targets.pop(0)
                    else:
                        continueRoutine = False
        
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trial"-------
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # save trial data
    trials.addData('response_times', response_times)
    trials.addData('trial_start_time', trial_start_time)
    
    # increment trial counter
    trial_counter += 1
    
    # store data for trials (TrialHandler)
    # the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed NUM_TRIALS repeats of 'trials'


# ------Prepare to start Routine "end_routine"-------
continueRoutine = True
routineTimer.add(5.000000)
# update component parameters for each repeat
# keep track of which components have finished
end_routineComponents = [text_end]
for thisComponent in end_routineComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
end_routineClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "end_routine"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = end_routineClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=end_routineClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_end* updates
    if text_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_end.frameNStart = frameN  # exact frame index
        text_end.tStart = t  # local t and not account for scr refresh
        text_end.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_end, 'tStartRefresh')  # time at next scr refresh
        text_end.setAutoDraw(True)
    if text_end.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text_end.tStartRefresh + 5-frameTolerance:
            # keep track of stop time/frame for later
            text_end.tStop = t  # not accounting for scr refresh
            text_end.frameNStop = frameN  # exact frame index
            win.timeOnFlip(text_end, 'tStopRefresh')  # time at next scr refresh
            text_end.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in end_routineComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "end_routine"-------
for thisComponent in end_routineComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
