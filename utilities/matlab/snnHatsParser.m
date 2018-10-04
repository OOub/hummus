% snnHatsParser.m

% Created by Omar Oubari 
% PhD - Institut de la Vision
% Email: omar.oubari@inserm.fr

% Last Version: 04/10/2018

% Information: snnHatsParser is a function that generates Histograms of averaged time surfaces (HATS) 
% from the scaled 64x56 n-Cars database and parses it so it can be fed into the Adonis spiking neural network simulator

% Dependencies: hats.m - load_atis_data.m

function [output, gridH] = snnHatsParser(folderPath, baseFileNames, r, tau, dt, spikeConversionRule, repetitions, timeBetweenPresentations, boolRandomisePresentationOrder)
    % folderPath - the path to the folder where all the recordings we want to parse are located

    % baseFileNames - the common name between all the files we want to feed
    % into the SNN so the dir method can locate them inside the folder
    % specified by the folderPath

    % r (optional) - radius
    
    % tau (optional) - decay of the time surface
    
    % dt (optional) - temporal window of the local memory time surface
    
    % spikeConversionRule (optional) - 1 to convert HATS to spikes via delays and
    % 2 to convert HATS to spikes via poissonian burts
    
    % repetitions (optional) - number of times each recording is presented
    
    % timeBetweenPresentations (optional) - time in microseconds between each
    % repetition
    
    % boolRandomisePresentationOrder (optional) - bool to select whether to
    % randomise the order of appeance of the recordings

    
    % handling optional arguments
    if nargin < 3
        r = 3;
        tau = 1e9;
        dt = 1e5;
        spikeConversionRule = 1;
        repetitions = 1;
        timeBetweenPresentations = 1000;
        boolRandomisePresentationOrder = false;
    elseif nargin < 4
        tau = 1e9;
        dt = 1e5;
        spikeConversionRule = 1;
        repetitions = 1;
        timeBetweenPresentations = 1000;
        boolRandomisePresentationOrder = false;
    elseif nargin < 5
        dt = 1e5;
        spikeConversionRule = 1;
        repetitions = 1;
        timeBetweenPresentations = 1000;
        boolRandomisePresentationOrder = false;
    elseif nargin < 6
        spikeConversionRule = 1;
        repetitions = 1;
        timeBetweenPresentations = 1000;
        boolRandomisePresentationOrder = false;
    elseif nargin < 7
        repetitions = 1;
        timeBetweenPresentations = 1000;
        boolRandomisePresentationOrder = false;
    elseif nargin < 8
        timeBetweenPresentations = 1000;
        boolRandomisePresentationOrder = false;
    elseif nargin < 9
        boolRandomisePresentationOrder = false;
    end
    
    % searching for all files fitting the description specified by the
    % folderPath and the baseFileNames
    datasetDirectory = dir(strcat(folderPath,'*',baseFileNames, '*'));
    H = {}; gridH = {}; spikeH = {};
    disp('files being parsed:')
    for i = 1:length(datasetDirectory)
        disp(strcat(datasetDirectory(i).folder, '/', datasetDirectory(i).name));
        data = load_atis_data(strcat(datasetDirectory(i).folder, '/', datasetDirectory(i).name));
        [H{i,1},gridH{i,1}] = hats([data.x, data.y, data.ts, data.p], r, tau, dt);
        temp = [];
        count = 0;
        for j = 1:size(gridH{i,1},1)
            for k = 1:size(gridH{i,1},2)
                % only making the active regions spike
                if gridH{i,1}(j,k) > 0
                    if spikeConversionRule == 1
                        % histogram value defines the timestamps of the spikes
                        % (higher values being closer to 0)
                        temp(end+1,:) = [1/(0.1*gridH{i,1}(j,k)), j-1, k-1];   
                    elseif spikeConversionRule == 2
                        % histogram value as poisson bursting activity
                        spikeNumber = round(gridH{i,1}(j,k))+1;
                        temp(end+1:end+spikeNumber,:) = [poissrnd(3, [spikeNumber 1])+count, repmat(j-1, [spikeNumber 1]), repmat(k-1, [spikeNumber 1])];
                    end
                    count = count+1;
                end
            end
        end
        spikeH{i,1} = sortrows(temp,1);
    end

    % generating the SNN input
    presentationOrder = repmat([1:length(spikeH)]',[repetitions 1]);
    
    if boolRandomisePresentationOrder == true
        presentationOrder = presentationOrder(randperm(size(presentationOrder,1)),:);
    end
    
    snnInput = []; spikeIntervals = [];
    for i = 1:length(presentationOrder)
        snnInput = [double(snnInput); double(spikeH{presentationOrder(i)}(:,1)), double(spikeH{presentationOrder(i)}(:,2)), double(spikeH{presentationOrder(i)}(:,3))];
        spikeIntervals(end+1,:) = length(snnInput(1:end,1));
    end

    % Shifting the timestamps so the presentations are sequential
    firstRowData = []; index = [];
    for i = 1:length(spikeH)
        firstRowData(end+1,:) = [spikeH{i}(1,1), spikeH{i}(1,2), spikeH{i}(1,3)];
        temp = find(snnInput(:,1) == firstRowData(end,1) & snnInput(:,2) == firstRowData(end,2) & snnInput(:,3) == firstRowData(end,3));
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
    
    output = struct('snnInput',snnInput,'spikeIntervals',spikeIntervals,'presentationOrder', presentationOrder);
end