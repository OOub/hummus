% snnPokerDataParser.m

% Created by Omar Oubari 
% PhD - Institut de la Vision
% Email: omar.oubari@inserm.fr

% Last Version: 04/09/2018

% Information: snnPokerDataParser is a function that parses the poker dataset (allcards.mat), so it can
% be fed into the Adonis spiking neural network simulator 

function [output, recordings] = snnPokerDataParser(allcards, repetitions, timeBetweenPresentations, timeJitter, conversionFactor, boolRandomisePresentationOrder, boolSpatialCrop, boolTemporalCrop, save)
    % allcards - path to the 'allcards.mat' file
    
    % repetitions - number of times each recording is presented
    
    % timeBetweenPresentations - time in microseconds between each
    % repetition
    
    % timeJitter (optional) - adds time jitter to the recording in microseconds
    
    % conversionFactor (optional) - to convert data from microseconds
    
    % boolRandomisePresentationOrder (optional) - bool to select whether to
    % randomise the order of appeance of the recordings
    
    % boolSpatialCrop (optional) - bool to select if we want to spatially crop the recordings

    % boolTemporalCrop (optional) - bool to select if we want to temporally crop the recordings
    
    % save (optional) - true to save the files, false otherwise
    
    % handling optional arguments
    if nargin < 4
        timeJitter = 0;
        conversionFactor = 10^-3;
        boolRandomisePresentationOrder = false;
        boolSpatialCrop = false;
        boolTemporalCrop = false; 
        save = true;
    elseif nargin < 5
        conversionFactor = 10^-3;
        boolRandomisePresentationOrder = false;
        boolSpatialCrop = false;
        boolTemporalCrop = false; 
        save = true;
    elseif nargin < 6
        boolRandomisePresentationOrder = false;
        boolSpatialCrop = false;
        boolTemporalCrop = false;  
        save = true;
    elseif nargin < 7
        boolSpatialCrop = false;
        boolTemporalCrop = false;
        save = true;
    elseif nargin < 8
        boolTemporalCrop = false;
        save = true;
    elseif nargin < 9
        save = true;
    end
    
    import = load(allcards);
    parsedData = import.ROI;
    
    % temporal crop
    if boolTemporalCrop == true
        prompt = {'Start Timestamp:','End Timestamp:'};
        title = 'Temporal Crop';
        dims = [1 35];
        definput = {'5000','8000'};
        answer = inputdlg(prompt,title,dims,definput);
        for i = 1:size(parsedData,1)
            for j = 1:size(parsedData,2)
                parsedData{i,j} = temporalCrop(parsedData{i,j}, str2double(answer{1}), str2double(answer{2}));
            end
        end
    end
    
    % spatial crop
    if boolSpatialCrop == true
        prompt = {'Starting Coordinate:','Square Size:'};
        title = 'Spatial Crop';
        dims = [1 35];
        definput = {'5','24'};
        answer = inputdlg(prompt,title,dims,definput);
        for i = 1:size(parsedData,1)
            for j = 1:size(parsedData,2)
                parsedData{i,j} = spatialCrop(parsedData{i,j}, str2double(answer{1}), str2double(answer{2}));
                parsedData{i,j}.Xaddress = parsedData{i,j}.Xaddress - str2double(answer{1});
                parsedData{i,j}.Yaddress = parsedData{i,j}.Yaddress - str2double(answer{1});
            end
        end
    end
    
    list = {'Clubs', 'Diamonds', 'Hearts', 'Spades'};
    pipsUsed = listdlg('ListString',list);
    
    prompt = {'Number Of Recordings For Each Pip (10 max):'};
    title = 'Number of Recordings';
    dims = [1 44];
    definput = {'1'};
    recordingNumber = inputdlg(prompt,title,dims,definput);
    
    if str2double(recordingNumber{1}) > size(parsedData,2)
        error(strcat('the number of recordings is too high. The database only contains',size(parsedData,2),'recordings for each pip'));
    end
    
    presentationOrder = repmat(pipsUsed',[repetitions 1]);
    presentationOrder = [presentationOrder,randi(str2double(recordingNumber{1}),[length(presentationOrder) 1])];
    
      if boolRandomisePresentationOrder == true
        presentationOrder = presentationOrder(randperm(size(presentationOrder,1)),:);
    end
    
    snnInput = []; spikeIntervals = [];
    for i = 1:length(presentationOrder)
        snnInput = [double(snnInput); double(parsedData{presentationOrder(i,1),presentationOrder(i,2)}.TimeStamp), double(parsedData{presentationOrder(i,1),presentationOrder(i,2)}.Xaddress), double(parsedData{presentationOrder(i,1),presentationOrder(i,2)}.Yaddress)];
        spikeIntervals(end+1,:) = length(snnInput(1:end,1));
    end
    
    % Shifting the Timestamps so the presentations are sequential
    setsUsed = unique(presentationOrder(:,2));
    
    firstRowData = []; index = [];
    for i = 1:length(pipsUsed)
        for j = 1:length(setsUsed)
            firstRowData(end+1,:) = [parsedData{pipsUsed(i),setsUsed(j)}.TimeStamp(1), parsedData{pipsUsed(i),setsUsed(j)}.Xaddress(1), parsedData{pipsUsed(i),setsUsed(j)}.Yaddress(1)];
        end
    end
    
    for i = 1:size(firstRowData,1)
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
    
    % adding time jitter
    if timeJitter > 0
        for i = 1:size(snnInput,1)
            jitter = normrnd(snnInput(i,1), timeJitter);
            while (jitter < 0)
                jitter = normrnd(snnInput(i,1), timeJitter);
            end
            snnInput(i,1) = jitter;
        end
        snnInput = sortrows(snnInput,1);
    end
    
    % converting from microseconds
    snnInput(:,1) = snnInput(:,1)*conversionFactor;
    
    for i = 1:size(spikeIntervals)
        presentationOrder(i,3) = snnInput(spikeIntervals(i),1);
    end

    output = struct('snnInput',snnInput,'spikeIntervals',spikeIntervals,'presentationOrder', presentationOrder);
    recordings = struct('originalData',import.ROI,'parsedData',parsedData);
    
    if save == true
        filename = strcat('pip', num2str(length(pipsUsed)),'_rep', num2str(repetitions), '_', 'jitter', num2str(timeJitter));
        dlmwrite(strcat(filename,'.txt'), snnInput, 'delimiter', ' ', 'precision', '%f');
        dlmwrite(strcat(filename,'Label.txt'), presentationOrder, 'delimiter', ' ', 'precision', '%f');
    end
end
        
function [output] = spatialCrop(ROI, startingCoordinate, squareSize)
    % only keeps events that occur within a square
    mask = and(and(ROI.Xaddress>startingCoordinate, ROI.Xaddress<=startingCoordinate+squareSize), and(ROI.Yaddress>startingCoordinate, ROI.Yaddress<=startingCoordinate+squareSize));
    fields = fieldnames(ROI);
    for i = 1:numel(fields)
        output.(fields{i}) = ROI.(fields{i})(mask);
    end
end

function [output] = temporalCrop(ROI, ts1, ts2)
    % only keeps events that have timestamps within [ts1, ts2]
    mask = and(ROI.TimeStamp >= ts1, ROI.TimeStamp <= ts2);
    fields = fieldnames(ROI);
    for i = 1:numel(fields)
        output.(fields{i}) = ROI.(fields{i})(mask);
    end
end