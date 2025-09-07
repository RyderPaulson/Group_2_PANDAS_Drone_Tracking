classdef CoTrackerSystem < matlab.System & matlab.system.mixin.CustomIcon
    % CoTracker3 MATLAB System Object with POI Tracking
    % System Object wrapper for integrating CoTracker 3 with POI tracking
    % for drone Simulink model.
    
    % Public, tunable properties
    properties(Nontunable)
        % Path to CoTracker model checkpoint (leave empty for default pretrained model)
        CheckpointPath = ''
        
        % Point of Interest coordinates (pixels). In the future this can be
        % set dynamicall but we haven't worked on that system yet. 
        POI_X = 640  % Default X coordinate for POI
        POI_Y = 360  % Default Y coordinate for POI
        
        % Enable POI tracking
        EnablePOIAtInit = true
    end
    
    properties(Nontunable, Logical)
        % Enable real-time processing mode (affects performance optimizations)
        RealTimeMode = true
    end
    
    % Private properties
    properties(Access = private)
        python_module
        initialized
        frame_count
        poi_set
    end

    %% Protected Methods
    methods(Access = protected)
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
            
            obj.initialized = false;
            obj.frame_count = 0;
            obj.poi_set = false;
            
            % Import Python module
            try
                obj.python_module = py.importlib.import_module('cotracker_matlab_interface');
                coder.extrinsic('fprintf');
                fprintf('Python CoTracker module loaded successfully\n');
            catch ME
                error('CoTrackerSystem:PythonImportFailed', ...
                    'Failed to load Python module: %s\nEnsure cotracker_matlab_interface.py is in Python path', ...
                    ME.message);
            end
            
            % Initialize CoTracker
            try
                checkpoint_path = py.None;
                if ~isempty(obj.CheckpointPath)
                    checkpoint_path = py.str(obj.CheckpointPath);
                end
                
                poi_x = py.None;
                poi_y = py.None;
                if obj.EnablePOIAtInit
                    poi_x = py.int(obj.POI_X);
                    poi_y = py.int(obj.POI_Y);
                end
                
                obj.initialized = obj.python_module.initialize_tracker(...
                    checkpoint_path, ...
                    poi_x, ...
                    poi_y);
                
                if obj.initialized
                    coder.extrinsic('fprintf');
                    fprintf('CoTracker initialized successfully in System Object\n');
                    if obj.EnablePOIAtInit
                        fprintf('POI set to (%d, %d)\n', obj.POI_X, obj.POI_Y);
                        obj.poi_set = true;
                    end
                else
                    error('CoTrackerSystem:InitializationFailed', 'Python interpreter indicated that CoTracker initialization failed');
                end
                
            catch ME
                error('CoTrackerSystem:InitializationError', 'Initialization failure: %s', ME.message);
            end
        end
        
        function annotated_frame = stepImpl(obj, frame)
            if ~obj.initialized
                error('CoTrackerSystem:NotInitialized', 'CoTracker not properly initialized');
            end
            
            % Validate input frame
            validateInputs(obj, frame);
            
            % Process frame with CoTracker
            try
                % Convert MATLAB array to Python numpy array
                frame_py = py.numpy.array(frame);
                
                % Process frame and get annotated result
                result_py = obj.python_module.process_single_frame(frame_py);
                
                % Convert result back to MATLAB
                annotated_frame = convertPythonArrayToMatlab(obj, result_py, size(frame));
                
                obj.frame_count = obj.frame_count + 1;
                
            catch ME
                error('CoTrackerSystem:ProcessingFailed', 'Failed to process frame: %s', ME.message);
            end
        end
        
        function resetImpl(obj)
            % Initialize / reset discrete-state properties
            
            if obj.initialized
                try
                    obj.python_module.reset_tracker();
                    obj.frame_count = 0;
                    obj.poi_set = obj.EnablePOIAtInit;
                    coder.extrinsic('fprintf');
                    fprintf('CoTracker System Object reset\n');
                catch ME
                    warning('CoTrackerSystem:ResetFailed', 'Failed to reset tracker: %s', ME.message);
                end
            end
        end
        
        function releaseImpl(obj)
            % Release resources, such as file handles
            
            % Reset tracker state
            if obj.initialized
                try
                    obj.python_module.reset_tracker();
                catch ME
                    warning('CoTrackerSystem:ReleaseFailed', 'Failed to release tracker resources: %s', ME.message);
                end
            end
            
            obj.initialized = false;
            obj.frame_count = 0;
            obj.poi_set = false;
        end
        
        function validateInputsImpl(obj, frame)
            % Validate inputs to the step method at initialization
            
            % Check frame dimensions and type
            if ~isa(frame, 'uint8')
                error('CoTrackerSystem:InvalidFrameType', ...
                    'Input frame must be of type uint8');
            end
            
            % Image must be colored
            if ndims(frame) ~= 3 || size(frame, 3) ~= 3
                error('CoTrackerSystem:InvalidFrameDimensions', ...
                    'Input frame must have dimensions H x W x 3 (RGB image)');
            end
            
            % Frame size validation
            if size(frame, 1) < 100 || size(frame, 2) < 100
                warning('CoTrackerSystem:SmallFrame', ...
                    'Frame size is very small (%dx%d). Tracking performance may be poor.', ...
                    size(frame, 1), size(frame, 2));
            end
        end
        
        function num = getNumInputsImpl(~)
            % Define total number of inputs for system with optional inputs
            num = 1;
        end
        
        function num = getNumOutputsImpl(obj)
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
        
        function out = isOutputFixedSizeImpl(obj)
            out = false;      % annotated_frame: variable size
        end
        
        function icon = getIconImpl(~)
            % Define icon for System block
            icon = matlab.system.display.Icon('.\media\Meta-Logo.png');
        end
        
        function name = getOutputNamesImpl(obj)
            % Return output port names for System block
            name = 'Annotated Frame';
        end
    end
    
    %% Public Methods for POI Control
    methods
        function success = setPOI(obj, x, y)
            % Set Point of Interest coordinates
            % Args:
            %   x: X coordinate in pixels
            %   y: Y coordinate in pixels
            % Returns:
            %   success: boolean indicating if POI was set successfully
            
            success = false;
            if obj.initialized
                try
                    result = obj.python_module.set_poi(py.int(x), py.int(y));
                    success = logical(result);
                    if success
                        obj.poi_set = true;
                        fprintf('POI set to (%d, %d)\n', x, y);
                    end
                catch ME
                    warning('CoTrackerSystem:SetPOIFailed', 'Failed to set POI: %s', ME.message);
                end
            else
                warning('CoTrackerSystem:NotInitialized', 'Tracker not initialized');
            end
        end
        
        function [position, visibility] = getCurrentPOIPosition(obj)
            % Get current POI position and visibility
            % Returns:
            %   position: [x, y] coordinates or empty if not available
            %   visibility: visibility score or NaN if not available
            
            position = [];
            visibility = NaN;
            
            if obj.initialized
                try
                    result = obj.python_module.get_poi_position();
                    pos_py = result{1};
                    vis_py = result{2};
                    
                    if ~isempty(pos_py) && ~isempty(vis_py)
                        position = double(pos_py);
                        visibility = double(vis_py);
                    end
                catch ME
                    warning('CoTrackerSystem:GetPOIFailed', 'Failed to get POI position: %s', ME.message);
                end
            end
        end
    end
    
    %% Private Methods
    methods(Access = private)
        function validateInputs(~, frame)
            % Raises error if the input image isn't a valid RGB image
            if ~isa(frame, 'uint8') || ndims(frame) ~= 3 || size(frame, 3) ~= 3
                error('CoTrackerSystem:InvalidInput', ...
                    'Frame must be uint8 array with dimensions H x W x 3');
            end
        end
        
        function matlab_array = convertPythonArrayToMatlab(~, python_array, target_size)
            % Convert Python numpy array to MATLAB array
            
            if nargin < 3
                target_size = [];
            end
            
            try
                % Convert to MATLAB array
                matlab_array = uint8(py.array.array('d', py.numpy.nditer(python_array)));
                
                if ~isempty(target_size)
                    matlab_array = reshape(matlab_array, target_size);
                end
                
            catch ME
                error('CoTrackerSystem:ConversionFailed', ...
                    'Failed to convert Python array to MATLAB: %s', ME.message);
            end
        end
    end
    
    methods(Access = protected)
        function s = saveObjectImpl(obj)
            % Set properties in structure s to values in object obj
            s = saveObjectImpl@matlab.System(obj);
            s.initialized = obj.initialized;
            s.frame_count = obj.frame_count;
            s.poi_set = obj.poi_set;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            % Set properties in object obj to values in structure s
            obj.initialized = s.initialized;
            obj.frame_count = s.frame_count;
            obj.poi_set = s.poi_set;
            
            % Call base class method
            loadObjectImpl@matlab.System(obj, s, wasLocked);
        end
    end
    
    methods(Static, Access = protected)
        function header = getHeaderImpl
            % Define header panel for System block dialog
            header = matlab.system.display.Header(mfilename('class'), ...
                'Title', 'CoTracker3 POI Tracker', ...
                'Text', ['Real-time point of interest tracking using Meta''s CoTracker3 model. ' ...
                'Tracks a specific point and returns annotated frames with red dot indicator.']);
        end
        
        function group = getPropertyGroupsImpl
            % Define property section(s) for System block dialog
            
            % Model Configuration
            modelGroup = matlab.system.display.Section(...
                'Title', 'Model Configuration', ...
                'PropertyList', {'CheckpointPath'});
            
            % POI Configuration
            poiGroup = matlab.system.display.Section(...
                'Title', 'Point of Interest Configuration', ...
                'PropertyList', {'POI_X', 'POI_Y', 'EnablePOIAtInit'});
            
            % System Configuration
            systemGroup = matlab.system.display.Section(...
                'Title', 'System Configuration', ...
                'PropertyList', {'RealTimeMode'});
            
            group = [modelGroup, poiGroup, systemGroup];
        end
    end
end