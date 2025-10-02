classdef TrackingSystem < matlab.System
    % CoTracker3 MATLAB System Object
    % Interfaces with Python implementation of the CoTracker algorithm to
    % track a point on the frame. 
    
    % Private properties
    properties(Access = private)
        py_module
        py_env
    end

    %% Protected Methods
    methods(Access = protected)
        function setupImpl(obj)
            % Load Python interpreter and import interface algorithm.

            % Run initial tracking

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
        
        function icon = getIconImpl(~)
            % Define icon for System block
            icon = matlab.system.display.Icon('.\media\Meta-Logo.png');
        end
        
        function name = getOutputNamesImpl(obj)
            % Return output port names for System block
            name = 'Annotated Video';
        end
    end
    
    %% Private Methods    
    methods(Static, Access = protected)
        function header = getHeaderImpl
            % Define header panel for System block dialog
            header = matlab.system.display.Header(mfilename('class'), ...
                'Title', 'CoTracker3 Tracker', ...
                'Text', ['Real-time point of interest tracking using Meta''s CoTracker3 model. ' ...
                'Tracks a specific point and returns annotated frames with red dot indicator.']);
        end
    end
end