% snn1DPatternGenerator.m

% Created by Omar Oubari 
% PhD - Institut de la Vision
% Email: omar.oubari@inserm.fr

% Last Version: 13/09/2018

% Information: snn1DPatternGenerator is a function that generates simple one dimensional patterns to be used for testing the Adonis spiking neural network simulator

function [output] = snn1DPatternGenerator(numberOfNeurons, numberOfPatterns, repetitions, patternMaxDuration, timeBetweenPresentations, timeJitter, boolRandomisePresentationOrder, boolSupervisedLearning)
    % numberOfNeurons - the number of neurons in the patterns being generated
    
    % numberOfPatterns - number of patterns to be generated
    
    % repetitions - number of times each recording is presented
    
    % patternMaxDuration (optional) - maximum duration of the patterns in milliseconds (actual
    % duration of each pattern is randomised to a value <= the maximum)
    
    % timeBetweenPresentations (optional) - time in milliseconds between each
    % repetition
    
    % timeJitter (optional) - adds time jitter to the recording. The value
    % is the standard deviation for the gaussian centered around a spike time 
    
    % boolRandomisePresentationOrder (optional) - bool to select whether to
    % randomise the order of appeance of the recordings
    
    % boolSupervisedLearning (optional) - creates a matrix of time and
    % neuronID to force certain neurons to fire at specific times (single layer support only)
    
    % handling optional arguments
    if nargin < 4
        patternMaxDuration = 20;
        timeBetweenPresentations = 100;
        timeJitter = 1;
        boolRandomisePresentationOrder = false;
        boolSupervisedLearning = false;
    elseif nargin < 5
        timeBetweenPresentations = 100;
        timeJitter = 1;
        boolRandomisePresentationOrder = false;
        boolSupervisedLearning = false;
    elseif nargin < 6
        timeJitter = 1;
        boolRandomisePresentationOrder = false;
        boolSupervisedLearning = false;
    elseif nargin < 7
        boolRandomisePresentationOrder = false;
        boolSupervisedLearning = false;
    elseif nargin < 8
        boolSupervisedLearning = false;
    end
    
    % generating the patterns
    patterns = cell(numberOfPatterns,1);
    for i = 1:numberOfPatterns
        neurons = 0:numberOfNeurons-1;
        neurons = neurons(:,randperm(size(neurons,2)))';
        time = sort(randperm(patternMaxDuration,numberOfNeurons)');
        time = time - time(1);
        patterns{i,1} = [time,neurons];
    end
    
    % generating the SNN input
    presentationOrder = repmat([1:numberOfPatterns]',[repetitions 1]);
    
    if boolRandomisePresentationOrder == true
        presentationOrder = presentationOrder(randperm(size(presentationOrder,1)),:);
    end
    
    snnInput = []; spikeIntervals = [];
    for i = 1:length(presentationOrder)
        snnInput = [snnInput; patterns{presentationOrder(i)}(:,1), patterns{presentationOrder(i)}(:,2)];
        spikeIntervals(end+1,:) = length(snnInput(1:end,1));
    end
    
    % Shifting the Timestamps so the presentations are sequential
    firstRowData = []; index = [];
    for i = 1:numberOfPatterns
        firstRowData(end+1,:) = [patterns{i}(1,1), patterns{i}(1,2)];
        temp = find(snnInput(:,1) == firstRowData(end,1) & snnInput(:,2) == firstRowData(end,2));
        index = [index;temp];
    end  
    index = sort(index);
    index(end+1) = length(snnInput)+1;
    
    index2 = index(1);
    for i = 2:length(index)
        if index(i) - index(i-1) > 1
            index2(end+1,:) = index(i);
        end
    end 
    
    for i = 3:size(index2,1)
        snnInput(index2(i-1):index2(i)-1,1) = snnInput(index2(i-1):index2(i)-1,1) + snnInput(index2(i-1)-1) + timeBetweenPresentations;
    end
    
    % adding time jitter
    if timeJitter > 0
        for i = 1:size(snnInput,1)
            jitter = normrnd(snnInput(i,1), timeJitter);
            while (jitter < 0)
                jitter = normrnd(snnInput(i,1), timeJitter);
            end
            snnInput(i,1) = jitter;
        end
    end
    
    % creating the teacher signal
    if boolSupervisedLearning == true
        responseNeurons = numberOfNeurons:numberOfNeurons+numberOfPatterns-1;
        count = 1;
        for i = 1:length(presentationOrder)
            % spike intervals as index + time in ms after the pattern as a desired
            teacherSignal(i,1) = snnInput(spikeIntervals(i)) + 5*count;
            count = count + 1;
            if mod(i,4) == 0
                count = 1;
            end
        end
    end
    
    if boolSupervisedLearning == true
        output = struct('snnInput',snnInput,'spikeIntervals',spikeIntervals,'presentationOrder', presentationOrder, 'teacherSignal', teacherSignal);
    else
        output = struct('snnInput',snnInput,'spikeIntervals',spikeIntervals,'presentationOrder', presentationOrder);
    end
end