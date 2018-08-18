import pymel.core as pm
def textOut():
    fileName = pm.fileDialog2(fileFilter='*.csv', caption='Write to a csv file')
    fileWrite = open(fileName[0], 'w')
    firstFrame = pm.intSlider('StartFrameSlider', query=True, value=True)
    lastFrame = pm.intSlider('EndFrameSlider', query=True, value=True)

    # animation time range
    animTimeRange = pm.checkBox('AnimBackRange_checkBox', query=True, value=True)
    if animTimeRange == True:
        animStart = pm.playbackOptions(query=True,animationStartTime=True)
        animEnd = pm.playbackOptions(query=True,animationEndTime=True) 
        firstFrame = int(animStart) 
        lastFrame = int(animEnd)   

    # playback time range
    playBackRange = pm.checkBox('PlayBackRange_checkBox', query=True, value=True)
    if playBackRange == True:
        playStart = pm.playbackOptions(query=True,minTime=True)
        playEnd = pm.playbackOptions(query=True,maxTime=True) 
        firstFrame = int(playStart) 
        lastFrame = int(playEnd)   

    DATA = []
    headerDATA = []


    # headerdata
    xValue = pm.checkBox('translateXcheckBox', query=True, value=True)  
    yValue = pm.checkBox('translateYcheckBox', query=True, value=True)  
    zValue = pm.checkBox('translateZcheckBox', query=True, value=True)  
    xRotate = pm.checkBox('rotateXcheckBox', query=True, value=True)  
    yRotate = pm.checkBox('rotateYcheckBox', query=True, value=True)  
    zRotate = pm.checkBox('rotateZcheckBox', query=True, value=True) 
    xScale = pm.checkBox('scaleXcheckBox', query=True, value=True)  
    yScale = pm.checkBox('scaleYcheckBox', query=True, value=True)  
    zScale = pm.checkBox('scaleZcheckBox', query=True, value=True)
    xShear = pm.checkBox('shearXcheckBox', query=True, value=True)  
    yShear = pm.checkBox('shearYcheckBox', query=True, value=True)  
    zShear = pm.checkBox('shearZcheckBox', query=True, value=True)  
    headerDATA.append("Object")
    headerDATA.append("Frame")
    # if xValue == True:
    headerDATA.append("X Trans")
    # if yValue == True:
    headerDATA.append("Y Trans")  
    # if zValue == True:
    headerDATA.append("Z Trans") 
    # if xRotate == True:
    # headerDATA.append("X Rotate")
    # if yRotate == True:
    headerDATA.append("Y Rotate")
    # if zRotate == True:
    # headerDATA.append("Z Rotate")
    # if xScale == True:
    # headerDATA.append("X Scale")
    # if yScale == True:
    # headerDATA.append("Y Scale")
    # if zScale == True:
    # headerDATA.append("Z Scale")
    # if xShear == True:
    # headerDATA.append("X Shear")
    # if yShear == True:
    # headerDATA.append("Y Shear")
    # if zShear == True:
    # headerDATA.append("Z Shear")


    finalheaderDATA = str(headerDATA).strip('[]')  
    selection = pm.ls(sl = True)
    print selection
    for j in range(0,1):
        fileWrite.write(finalheaderDATA + "\n")
        for i in range(firstFrame,lastFrame + 1):
            pm.currentTime(i)
            new_message = 'drum.' + str(i).zfill(4) + '.jpg'
            for object in selection:
                # TRANSLATION

                transX = pm.getAttr(object + '.translateX')
                DATA.append(float("{0:.3f}".format(transX)))

                transY = pm.getAttr(object + '.translateY')
                DATA.append(float("{0:.3f}".format(transY)))

                transZ = pm.getAttr(object + '.translateZ')
                DATA.append(float("{0:.3f}".format(transZ)))
                # ROTATION
                  
                yRotateValue = pm.getAttr(object + '.rotateY')
                DATA.append(float("{0:.3f}".format(yRotateValue)))
                       
                finalDATA = str(DATA).strip('[]')
                finalDATA = new_message + ',' + str(i) + ',' + str(finalDATA) + "\n"
                fileWrite.write(finalDATA)
                DATA = []
        fileWrite.close()