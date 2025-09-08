% Need to wait until core CoTracker implementation is finished to do. 

classdef CoTrackerSystem < matlab.System
    % CoTracker3 MATLAB System Object
    % Interfaces with Python implementation of the CoTracker algorithm to
    % track a point on the frame. 
    
    % Public, tunable properties
    properties(Nontunable)
        % Query coordinates should be set automatically in the final
        % system. Here they will be defined arbitrarily.
        query
    end
    
    % Private properties
    properties(Access = private)
        python_module
    end

    %% Protected Methods
    methods(Access = protected)
        function setupImpl(obj)
            % Load Python interpreter and import interface algorithm.

            % Recieve inputed query point. And set it.

            % Perform any other necessary startup steps for the co-tracker
            % algorithm. 
        end
        
        function annotated_frame = stepImpl(obj, frame)
            % For now, passes through frame.
            annotated_frame = frame;
        end
        
        function num = getNumInputsImpl(~)
            % Define total number of inputs for system with optional inputs
            num = 1;
        end
        
        function num = getNumOutputsImpl(~)
            % Define the total number of output ports
            num = 1;
        end
        
        function out = getOutputSizeImpl(obj)
            % Return size for each output port
            out = [1080, 1920, 3];  % annotated_frame: variable height/width, 3 channels
        end
        
        function out = getOutputDataTypeImpl(obj)
            % Return data type for each output port
            out = 'uint8';    % annotated_frame
        end
        
        function icon = getIconImpl(~)
            % Define icon for System block
            icon = matlab.system.display.Icon('.\media\Meta-Logo.png');
        end
        
        function name = getOutputNamesImpl(obj)
            % Return output port names for System block
            name = 'Annotated Video';
        end
    end
    
    %% Public Methods for Query Control
    methods
        function success = setQuery(obj, x, y)
            % Set Point of Interest coordinates
            % Args:
            %   x: X coordinate in pixels
            %   y: Y coordinate in pixels
            % Returns:
            %   success: boolean indicating if query was set successfully
            
            success = false;
        end
        
        function [position] = getCurrentPOIPosition(obj)
            % Get current POI position and visibility
            % Returns:
            %   position: [x, y] coordinates or empty if not available
            
        end
    end
    
    %% Private Methods    
    methods(Static, Access = protected)
        function header = getHeaderImpl
            % Define header panel for System block dialog
            header = matlab.system.display.Header(mfilename('class'), ...
                'Title', 'CoTracker3 POI Tracker', ...
                'Text', ['Real-time point of interest tracking using Meta''s CoTracker3 model. ' ...
                'Tracks a specific point and returns annotated frames with red dot indicator.']);
        end
    end
end