% snnAnalysis.m

% Created by Omar Oubari 
% PhD - Institut de la Vision
% Email: omar.oubari@inserm.fr

% Last Version: 19/11/2018

% Information: snnAnalysis is a function that analyses the output of a
% spiking neural network built using the Adonis simulator

% Dependencies: snnReader.m

function [accuracy, actualLabels, predictedLabels] = snnAnalysis(testOutputLogger, labels)
    data = snnReader(testOutputLogger, 2);
    testLabels = importdata(labels);
    testLabels.data = testLabels.data+data.learningOffSignal+1000;
    
    % Creating the actual and predicted label matrices
    outputNeurons = unique(data.outputSpikes.postN);
    classes = unique(testLabels.textdata);
    
    if length(outputNeurons) ~= length(classes)
        error('the number of neurons in the last layer is different from the number of classes found in the labels')
    end
    
    permutations = perms(1:length(classes));
    actualLabels = zeros(length(testLabels.data), 1);
    predictedLabels = zeros(length(testLabels.data), length(permutations));
    
    for i = 1:length(testLabels.data)
        [c, idx] = min(abs(data.outputSpikes.timestamp-testLabels.data(i)));
        if c < 900
            temp = data.outputSpikes.postN(idx);
        else
            temp = NaN;
        end
        
        for j = 1:length(classes)
            if temp == outputNeurons(j)
                for k = 1:length(permutations)
                    predictedLabels(i, k) = permutations(k, j);
                end
            elseif isnan(temp)
                for k = 1:length(permutations)
                    predictedLabels(i, k) = NaN;
                end
            end
            if testLabels.textdata{i} == classes{j}
                actualLabels(i, :) = j;
            end
        end
    end
    
    % Accuracy
    for i = 1:length(permutations)
        comparison = actualLabels == predictedLabels(:,i);
        accuracy(i,:) = (sum(comparison==1)/length(actualLabels))*100;
    end
    
    [accuracy, idx] = max(accuracy);
    predictedLabels = predictedLabels(:, idx);
end