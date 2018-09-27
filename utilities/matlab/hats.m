% hats.m

% Created by Siohoi Ieng and Omar Oubari
% Institut de la Vision

% Emails: 
% siohoi.ieng@inserm.fr
% omar.oubari@inserm.fr

% Last Version: 27/09/2018

% Information: hats calculates the histogram of averaged times surfaces specifically for the n-Cars database scaled to 64x56

function [H, gridH] = hats(data, r, tau, dt, displayHistogram)
    % data: 4-column array: [x y t p]
    
    % r: radius
    
    % tau: decay of the time surface
    
    % dt: temporal window of the local memory time surface

    % displayHistogram: uses the imagesc function to display the histogram
    % organised as a spatial grid
    
    
    % handling optional arguments
    if nargin < 2
        r = 3;
        tau = 1e9;
        dt = 1e5;
        displayHistogram = false;
    elseif nargin < 3
        tau = 1e9;
        dt = 1e5;
        displayHistogram = false;
    elseif nargin < 4
        dt = 1e5;
        displayHistogram = false;
    elseif nargin < 5
        displayHistogram = false;
    end
    
    % scaled n-Cars database size
    Width = 64;
    Height = 56;

    % Cell size
    cellW = floor(Width/10);
    cellH = floor(Height/10); 

    % histogram hc size
    hc = 2*r+1;

    timeSurface = zeros(hc,hc);
    H = zeros(cellH*cellW*hc*hc,1);

    for i = 1:size(data,1)
        if (data(i,1) > r && data(i,1) <= cellW*10) && (data(i,2) > r && data(i,2) <= cellH*10)

            % reject events that are not in cells: this part needs to be more generic
            if (data(i,1)>2 && data(i,1)<Width-2 && data(i,2)>3 && data(i,2)<Height-3)

                % getting the correct cell for a 5-line matrix structure numbered from top to down, left to right
                cellID = cellH*floor((data(i,1)-r-1)/10)+floor((data(i,2)-r-1)/10)+1;

                % finding the right event indices
                lst = find(abs(data(1:i-1,1)-data(i,1))<=r & abs(data(1:i-1,2)-data(i,2))<=r & (data(i,3)-data(1:i-1,3))<=dt & data(1:i-1,4)==data(i,4));

                if ~isempty(lst)
                    for j = 1:size(lst,1)
                        % computing the time surface
                        timeSurface(data(lst(j),2)-data(i,2)+r+1,data(lst(j),1)-data(i,1)+r+1) = timeSurface(data(lst(j),2)-data(i,2)+r+1,data(lst(j),1)-data(i,1)+r+1) + exp(-(data(i,3)-data(lst(j),3))/tau);
                    end

                    % summing time surfaces into histograms and normalising by the number of events
                    H((cellID-1)*hc^2+1:cellID*hc^2) = H((cellID-1)*hc^2+1:cellID*hc^2) + timeSurface(:) / length(lst);

                    % resetting time surfaces for next cell
                    timeSurface = zeros(hc,hc);
                end
            end  
        end
    end
  
    temp = reshape(H, [hc^2 length(H)/hc^2]);
    for i = 1:size(temp,2)
        cells{i,1} = reshape(temp(:,i), [hc hc])'; 
    end
    cells = reshape(cells, [cellW cellH])';

    gridH = zeros(hc*cellW, hc*cellH);
    for i = 1:size(cells,1)
        icount = hc*i;
        for j = 1:size(cells,2)
            jcount = hc*j;
            gridH(icount-(hc-1):icount,jcount-(hc-1):jcount) = cells{i,j};
        end
    end
    
    if displayHistogram == true
        imagesc(gridH)
    end
end

