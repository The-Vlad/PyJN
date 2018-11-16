def fit_on_batches(model, x_data, y_data, mode=None, epochs=1, batch_size=32, validation_split=0., data_split=1., target_img_size=None, patience=None, verbose = 1):
    
    import os
    import time
    import numpy as np
    from keras.preprocessing import image
    np.random.seed(42)

    if mode==None:
        mode='img' if target_img_size!=None else 'arr'

    if mode!='arr':
        x_data, y_data = [x_data + i for i in os.listdir(x_data)], [y_data + i for i in os.listdir(y_data)]

    def imgInPath(path):
        return image.img_to_array(image.load_img(path, target_size=target_img_size))
    def maskInPath(path):
        return image.img_to_array(image.load_img(path, target_size=target_img_size, color_mode = "grayscale"))

    start_fit_time = time.clock()
    maxDataSplit = int(len(x_data) * data_split)
    maxValSplit = int(maxDataSplit * validation_split)
    dataXArr = np.random.choice(len(x_data), maxDataSplit, replace=False)
    trainXArr = dataXArr[maxValSplit:]
    valXArr = dataXArr[:maxValSplit]
    xTrain = np.array([x_data[i] for i in trainXArr])
    yTrain = np.array([y_data[i] for i in trainXArr])
    xVal = np.array([x_data[i] for i in valXArr])
    yVal = np.array([y_data[i] for i in valXArr])

    if len(xVal)==0:
        print('Train on {} samples:'.format(len(xTrain)))
    else:
        print('Train on {} samples, validate on {} samples:'.format(len(xTrain), len(xVal)))

    maxCtrlAcc = 0
    patCombo = 0
    ctrlInfo = None
    ETAE = 0
    hystory = {}
    hystory['BE'] = 0
    for epoch in range(epochs):
        if verbose==1: print('Epoch {}/{}'.format(epoch+1, epochs))
        start_epoch_time = time.clock()
        outInfo = []
        if len(xTrain) > 0:
            batch=0
            startX=0
            endX=0
            ETA=0
            metricsSum=[0 for i in range(len(model.metrics_names))]
            while endX < len(xTrain):
                start_train_time = time.clock()
                batch += 1
                startX = batch_size*(batch-1)
                if batch_size*batch > len(xTrain):
                    endX = len(xTrain)
                else:
                    endX = batch_size*batch

                if mode=='arr':
                    x_batch = np.array([i for i in xTrain[startX:endX]])
                    y_batch = np.array([i for i in yTrain[startX:endX]])
                if mode=='img':
                    x_batch = np.array([imgInPath(i) for i in xTrain[startX:endX]])
                    y_batch = np.array([imgInPath(i) for i in yTrain[startX:endX]])
                    x_batch = x_batch.astype('float32')
                    y_batch = y_batch.astype('float32')
                    x_batch /= 255
                    y_batch /= 255
                if mode=='mask':
                    x_batch = np.array([imgInPath(i) for i in xTrain[startX:endX]])
                    y_batch = np.array([maskInPath(i) for i in yTrain[startX:endX]])
                    x_batch = x_batch.astype('float32')
                    y_batch = y_batch.astype('float32')
                    x_batch /= 255
                    y_batch /= 255

                #hystory = model.fit(x_batch, y_batch, epochs=1, batch_size=batch_size, verbose=0)
                #metrics = [i[0] for i in list(hystory.history.values())]
                metrics = model.train_on_batch(x_batch, y_batch)
                metricsSum = [metricsSum[i]+metrics[i] for i in range(len(metrics))]

                if len(xTrain) > 1 and verbose==1:
                    delta_train_time = (time.clock()-start_train_time)/batch_size
                    iETA = delta_train_time*(len(xTrain)-endX)
                    ETA = int(iETA+(ETA-iETA)*0.1)
                    print('Training batches: {:.2f}% - ETA: {:<5}'.format(endX/len(xTrain)*100, str(ETA)+'s'), end='\r', sep='')

            avgMetrics = [i/batch for i in metricsSum]
            metricsInfo = list(zip(model.metrics_names, avgMetrics))
            for i in range(len(metricsInfo)):
                outInfo.append(' | {}: {:.4f}'.format(metricsInfo[i][0], metricsInfo[i][1]))

        if len(xVal) > 0:
            valBatch=0
            startValX=0
            endValX=0
            ETA=0
            evalSum=[0 for i in range(len(model.metrics_names))]
            while endValX < len(xVal):
                start_val_time = time.clock()
                valBatch += 1
                startValX = batch_size*(valBatch-1)
                if batch_size*valBatch > len(xVal):
                    endValX = len(xVal)
                else:
                    endValX = batch_size*valBatch

                if mode=='arr':
                    x_val_batch = np.array([i for i in xVal[startValX:endValX]])
                    y_val_batch = np.array([i for i in yVal[startValX:endValX]])
                if mode=='img':
                    x_val_batch = np.array([imgInPath(i) for i in xVal[startValX:endValX]])
                    y_val_batch = np.array([imgInPath(i) for i in yVal[startValX:endValX]])
                    x_val_batch = x_val_batch.astype('float32')
                    y_val_batch = y_val_batch.astype('float32')
                    x_val_batch /= 255
                    y_val_batch /= 255
                if mode=='mask':
                    x_val_batch = np.array([imgInPath(i) for i in xVal[startValX:endValX]])
                    y_val_batch = np.array([maskInPath(i) for i in yVal[startValX:endValX]])
                    x_val_batch = x_val_batch.astype('float32')
                    y_val_batch = y_val_batch.astype('float32')
                    x_val_batch /= 255
                    y_val_batch /= 255

                evaluate = model.evaluate(x_val_batch, y_val_batch, verbose=0)
                evalSum = [evalSum[i]+evaluate[i] for i in range(len(evaluate))]

                if len(xVal) > 1 and verbose==1:
                    delta_val_time = (time.clock()-start_val_time)/batch_size
                    iETA = delta_val_time*(len(xVal)-endValX)
                    ETA = int(iETA+(ETA-iETA)*0.05)
                    print('Validation batches: {:.2f}% - ETA: {:<5}'.format(endValX/len(xVal)*100, str(ETA)+'s'), end='\r', sep='')

            avgEval = [i/valBatch for i in evalSum]
            evalInfo = list(zip(['val_'+i for i in model.metrics_names], avgEval))
            for i in range(len(evaluate)):
                outInfo.append(' | {}: {:.4f}'.format(evalInfo[i][0], evalInfo[i][1]))

        if verbose==1:
            outInfo.insert(0, ' - {:<5}'.format(str(int(time.clock()-start_epoch_time))+'s'))
            outInfo = ''.join(outInfo)
            print(outInfo)
            
        if verbose==0:
            delta_epoch_time = time.clock()-start_epoch_time
            iETAE = delta_epoch_time*(epochs-epoch+1)
            ETAE = int(iETAE+(ETAE-iETAE)*0.05)
            shortOutInfo = outInfo[1::2]
            shortOutInfo = ''.join(shortOutInfo)
            print('Epoch {}{} - ETA: {:<6}'.format(str(epoch+1)+'/'+str(epochs), shortOutInfo, str(ETAE)+'s'), end='\r', sep='')
            
        ctrlInfo = metricsInfo if len(xVal) == 0 else metricsInfo+evalInfo

        if len(hystory)==1:
            for x, y in ctrlInfo:
                hystory[x] = [y]
        else:
            for x, y in ctrlInfo:
                hystory[x].append(y)

        if patience!=None:
            if patience>=0:
                if ctrlInfo[-1][1] >= maxCtrlAcc:
                    if ctrlInfo[-1][1] > maxCtrlAcc:
                        patCombo = -1
                    maxCtrlAcc = ctrlInfo[-1][1]
                    bestModelInfo = [model, epoch]
                if ctrlInfo[-1][1] <= maxCtrlAcc:
                    patCombo += 1

                if patCombo >= patience:
                    print('Retraining, stop learning... Saved the best epoch -', bestModelInfo[1]+1)
                    model = bestModelInfo[0]
                    hystory['BE'] = bestModelInfo[1]+1
                    break

            if patience==-1:
                if ctrlInfo[-1][1] >= maxCtrlAcc:
                    maxCtrlAcc = ctrlInfo[-1][1]
                    bestModelInfo = [model, epoch]

                if (epoch+1==epochs):
                    if bestModelInfo[1]!=epoch:
                        print('Saved the best epoch -', bestModelInfo[1]+1)
                    model = bestModelInfo[0]
                    hystory['BE'] = bestModelInfo[1]+1
    if verbose==0:
        outInfo.insert(0, ' - {:<6}'.format(str(int(time.clock()-start_fit_time))+'s'))
        outInfo = ''.join(outInfo)
        print(outInfo)
    return hystory