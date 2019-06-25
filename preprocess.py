def analyse(auto):
    resTarget = []
    resDist = []
    startA = auto.start
    for dst in auto.grf[startA]:
        target = auto.grf[startA][dst][0]["target"]
        distance = auto.grf[startA][dst][0]["distance"]
        resTarget.append(target)
        resDist.append(distance)
        ##assuming all the distance restriction are the same
    return resTarget, resDist[0]