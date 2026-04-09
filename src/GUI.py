import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import csv
import numpy as np
import rasterio
from rasterio.plot import show
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import cm
from matplotlib.colors import Normalize
from pyproj import Transformer
import matplotlib.image as mpimg
from API_caller import download_dem_for_samples
from matplotlib.ticker import MaxNLocator, ScalarFormatter

tif_path = None
metadata_csv_path = None
loaded_sample_metadata = []


dtif_path = None #file to be loaded from.
metadata_csv_path = None #file containing sample metadata.
loaded_sample_metadata = [] #store metadata loaded from CSV.

def start_gui(run_program): #entry point for the program.

    def set_coordinate_entries(lon, lat):
        row_index = selected_row_var.get() - 1 #work out which sample row should be auto-filled.

        if not (0 <= row_index < len(sample_entries)):
            return

        lon_entry, lat_entry, _, _ = sample_entries[row_index]

        lon_entry.delete(0, tk.END) #delete text from longitude box.
        lon_entry.insert(0, f"{lon:.6f}") #insert given longitude with 6 decimal digits of precision.
        
        lat_entry.delete(0, tk.END) #delete text from latitude box.
        lat_entry.insert(0, f"{lat:.6f}") #insert given latitude with 6 decimal digits of precision.

    def add_scale_bar(ax, length_m: float) -> None:
        x0, x1 = ax.get_xlim() #retrieve x limits of axes.
        y0, y1 = ax.get_ylim() #retrieve y limits of axes.

        x = x0 + (x1 - x0) * 0.07
        y = y0 + (y1 - y0) * 0.07 #place bar 7% up and to the right from the bottom left corner.

        ax.plot([x, x + length_m], [y, y], linewidth=4, color="black") #draw a horizontal line.
        label = f"{int(length_m)} m" if length_m < 1000 else f"{length_m / 1000:.1f} km" #label line in metres if below 1km or else kilometres.
        ax.text(
            x + length_m / 2.0,
            y + (y1 - y0) * 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
        ) #styling for label.

    class RightSideBar(ttk.Frame): #right side bar showing the DEM preview and overlay result, ttk.Frame specifies it as a widget container.
        def __init__(self, parent):
            super().__init__(parent, padding=8) #initialise the object as a ttk.Frame and add padding.


            self.dem = None #store DEM data.
            self.dem_transform = None #store raster transformer.
            self.dem_crs = None #store raster CRS.
            self.dem_path = None #store file path to DEM.
            self.count_overlay = None #store frequency count result.
            self.observer_points_xy = [] #store observer cooridnates.

            self.point_selected_callback = None #reference to helper function for auto filling boxes on click.
            self.clicked_points = [] #store the coordinates the user has clicked on.
            self.tip = None

            self.rowconfigure(1, weight=1) #only allow preview to grow if needed.
            self.columnconfigure(0, weight=1) #allow preview section to be streched sideways.

            self.title_label = ttk.Label(self, text="DEM PREVIEW", font=("Segoe UI", 12, "bold")) #add Title.
            self.title_label.grid(row=0, column=0, sticky="w", pady=(0, 6)) #position Title.

            self.fig = Figure(figsize=(11, 5.8), dpi=100) #create a Matplotlib Figure object.
            self.ax_count = self.fig.add_subplot(111) #add axes.
            self.ax_count.set_title("No DEM loaded") #if no DEM loaded, inform user.

            self.scale_bar_length_m = None #store scale bar length for result view.

            self.ax_count.set_xticks([])
            self.ax_count.set_yticks([]) #do not show ticks around empty canvas message.

            self.canvas = FigureCanvasTkAgg(self.fig, master=self) #create a Tkinter compatible canvas wrapper for Matplotlib.
            self.canvas_widget = self.canvas.get_tk_widget() #render it with equivalent widget corresponding to a Matplotlib figure.
            self.canvas_widget.grid(row=1, column=0, sticky="nsew") #force DEM canvas to fill entire grid cell.

            self.toolbar_frame = ttk.Frame(self) #add container for Matplotlib toolbar.
            self.toolbar_frame.grid(row=2, column=0, sticky="ew") #position Matplotlib toolbar.
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame, pack_toolbar=False) #create the Matplotlib toolbar to add interactivity to the canvas.
            self.toolbar.update() #standard practice: refresh toolbar before displaying.
            self.toolbar.pack(side="left") #move toolbar to the left.
            
            self.canvas.mpl_connect("button_press_event", self.on_click)
            self.canvas.mpl_connect("scroll_event", self.on_scroll)

            self.show_overlay = tk.BooleanVar(value=True) #create a Boolean state for whether overlay should be visible, default to show.
            self.count_overlay_im = None #default to no overlay image.
            self.count_overlay_cbar = None #default to no colour bar.

            self.view_xlim = None
            self.view_ylim = None #store information about current manual zoom so it isn't lost on refresh.
            self.view_extent = None #store automatic zoom calculated by programw for the current result.

            menubar = tk.Menu(parent) #create menu for the window in the top bar...
            parent.config(menu=menubar) #...and attach it.

            file_menu = tk.Menu(menubar, tearoff=0) #create file menu inside the top bar...
            menubar.add_cascade(label="File", menu=file_menu) #...and add it.
            file_menu.add_command(label="Load metadata CSV", command=load_metadata_file) #create a button to load metadata from CSV.

            view_menu = tk.Menu(menubar, tearoff=0) #create view menu inside the top bar...
            menubar.add_cascade(label="View", menu=view_menu) #...and add it.

            view_menu.add_checkbutton(
                label="Show overlay",
                variable=self.show_overlay,
                command=self.toggle_overlay
            ) #create a button to toggle overlay.

            self.canvas.mpl_connect("button_press_event", self.on_click) #create a click handler to detect what coordinates a user may click on when drawing a polygon.
  
        def on_scroll(self, event): #when the user scrolls mouse wheel over the plot.
            if event.inaxes != self.ax_count:
                return #if scroll happens outside the plot, do nothing.

            if event.xdata is None or event.ydata is None:
                return #if graph coordinates cannot be worked out, do nothing.

            if self.toolbar.mode != "":
                return #if a Matplotlib tool is active, do not interfere.

            xdata = event.xdata #x position of mouse in data coordinates.
            ydata = event.ydata #y position of mouse in data coordinates.

            cur_xlim = self.ax_count.get_xlim() #retrieve current x axis limits.
            cur_ylim = self.ax_count.get_ylim() #retrieve current y axis limits.

            x_left = xdata - cur_xlim[0] #distance from mouse to left edge.
            x_right = cur_xlim[1] - xdata #distance from mouse to right edge.
            y_bottom = ydata - cur_ylim[0] #distance from mouse to bottom edge.
            y_top = cur_ylim[1] - ydata #distance from mouse to top edge.

            if event.button == "up":
                scale_factor = 0.8 #zoom in.
            elif event.button == "down":
                scale_factor = 1.25 #zoom out.
            else:
                return #if unknown scroll direction, do nothing.

            new_xlim = [xdata - x_left * scale_factor, xdata + x_right * scale_factor]
            new_ylim = [ydata - y_bottom * scale_factor, ydata + y_top * scale_factor] #scale limits around the mouse position.

            self.ax_count.set_xlim(new_xlim)
            self.ax_count.set_ylim(new_ylim)

            self.view_xlim = self.ax_count.get_xlim()
            self.view_ylim = self.ax_count.get_ylim() #store manual zoom so redraws keep it.

            self.canvas.draw_idle() #refresh canvas.    

        def format_axes_nicely(self, ax) -> None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

            x_formatter = ScalarFormatter(useMathText=True)
            y_formatter = ScalarFormatter(useMathText=True)

            x_formatter.set_scientific(True)
            y_formatter.set_scientific(True)

            x_formatter.set_powerlimits((0, 0))
            y_formatter.set_powerlimits((0, 0))

            ax.xaxis.set_major_formatter(x_formatter)
            ax.yaxis.set_major_formatter(y_formatter)

            ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)

        def on_click(self, event): #when the user clicks mouse.
            if self.dem is None:
                return
            if event.inaxes != self.ax_count:
                return #if click happens outside of canvas, do nothing.
            if event.xdata is None or event.ydata is None:
                return #if the graph coordinates cannot be worked out, do nothing.
            if self.toolbar.mode != "":
                return

            x = event.xdata #obtain x coordinate user clicked on.
            y = event.ydata #obtain y coordinate user clicked on.
            self.clicked_points.append((x, y)) #return coordinates user clicked on.

            if self.dem_crs is not None and self.point_selected_callback is not None: #if a coordinate system can be found, and the helper function is ready.
                transformer = Transformer.from_crs(self.dem_crs, "EPSG:4326", always_xy=True) #create DEM transformer.
                lon, lat = transformer.transform(x, y) #convert clicked position into longitude and latitude.
                self.point_selected_callback(lon, lat) #send coordinates to helper function.

            self._redraw()

        def load_dem(self, dem_path):
            with rasterio.open(dem_path) as src: #open GeoTIFF file.
                arr = src.read(1).astype(np.float64) #read elevation raster band.
                nodata = src.nodata #value that raster uses for "nodata".

                if nodata is not None: #if there are cells that have no data.
                    arr = np.ma.masked_equal(arr, nodata) #mask each one.

                arr = np.ma.masked_invalid(arr) #mask invalid cells.       
                arr = np.ma.masked_where(np.abs(arr) > 1e20, arr) #mask cells with an absurdly large size.

                self.dem = arr #store DEM array in GUI.
                self.dem_transform = src.transform #store affine transform.
                self.dem_crs = src.crs #store raster coordinate system.
                self.dem_path = dem_path #store DEM path.

            self.count_overlay = None #clear previous DEM overlay.
            self.observer_points_xy = [] #clear previous observer points.
            self.scale_bar_length_m = None #clear previous scale bar.
            self._redraw() #redraw DEM.

        def set_results(self, count_overlay, observer_points=None, view_extent=None, scale_bar_length_m=None):
            if self.dem is None:
                raise ValueError("Load a DEM before setting an overlay.") #check a DEM file is present.

            self.count_overlay = count_overlay #store DEM LoS render.
            self.observer_points_xy = observer_points if observer_points is not None else [] #store observer coordinates.
            self.view_extent = view_extent #store automatic zoom for the current result.
            self.scale_bar_length_m = scale_bar_length_m #store scale bar length for the current result.
            self.view_xlim = None
            self.view_ylim = None #reset manual zoom so a fresh submit uses the program's automatic zoom.
            self._redraw() #render.


        def toggle_overlay(self): #toggle overlay on/off.
            self._redraw() #rerender.


            if self.count_overlay_im is not None: #if overlay exists.
                self.count_overlay_im.set_visible(show) #toggle on/off.

            if self.count_overlay_cbar is not None: #if colourbar exists.
                self.count_overlay_cbar.ax.set_visible(show) #toggle on/off.

            self.canvas.draw_idle() #update display.

        def clear_overlay(self):
            self.count_overlay = None #remove any previous overlay.
            self.observer_points_xy = [] #remove any previous observer point.
            self.scale_bar_length_m = None #remove previous scale bar information.
            self.view_extent = None #remove previous automatic zoom information.
            self._redraw() #render preview area again.

        def _remove_colourbars(self):

            if self.count_overlay_cbar is not None:
                self.count_overlay_cbar.remove()
                self.count_overlay_cbar = None

        def _redraw(self):
            self._remove_colourbars()
            self.ax_count.clear() #clean axes to prevent buildup.

            if self.dem is None: #if no DEM given, show blank DEM message.
                self.ax_count.set_title("No DEM loaded")
                self.ax_count.set_xticks([])
                self.ax_count.set_yticks([])
                self.canvas.draw_idle()
                return

            if self.count_overlay is not None and self.show_overlay.get() and self.view_extent is not None: #if a result exists and overlays are enabled.
                left, right, bottom, top = self.view_extent

                count_vmax = max(1, int(np.max(self.count_overlay))) #find biggest count to scale colourbar.

                self.count_overlay_im = self.ax_count.imshow(
                    self.count_overlay,
                    extent=(left, right, bottom, top),
                    origin="upper",
                    cmap="viridis",
                    vmin=0,
                    vmax=count_vmax,
                ) #render the frequency raster exactly like visibility_frequency.py.

                self.count_overlay_cbar = self.fig.colorbar(self.count_overlay_im, ax=self.ax_count)
                self.count_overlay_cbar.set_label("Number of observers seeing each cell")

                self.ax_count.set_title("Visibility frequency")
                self.ax_count.set_xlabel("X")
                self.ax_count.set_ylabel("Y")
                self.ax_count.set_aspect("equal")
                self.format_axes_nicely(self.ax_count)

                if self.scale_bar_length_m is not None:
                    add_scale_bar(self.ax_count, self.scale_bar_length_m) #draw scale bar exactly like visibility_frequency.py.

            else:
                show(
                    self.dem,
                    transform=self.dem_transform,
                    ax=self.ax_count,
                    cmap="terrain"
                )  #render DEM base image.

                self.ax_count.set_title("DEM PREVIEW")
                self.ax_count.set_xlabel("Easting (m)")
                self.ax_count.set_ylabel("Northing (m)")
                self.count_overlay_im = None

            if self.view_xlim is not None and self.view_ylim is not None:
                self.ax_count.set_xlim(self.view_xlim)
                self.ax_count.set_ylim(self.view_ylim) #redraw zoom based on saved information.

            self.fig.tight_layout()
            self.canvas.draw_idle()

        def hide_tip(self, event=None):
            if self.tip is not None:
                self.tip.destroy() #remove window.
                self.tip = None #...and the reference to it.
                
    class LeftSideBar: #for each helper button

            def __init__(self, widget, text):
                self.widget = widget #widget the popup belongs to.
                self.text = text #text that should appear in popup.
                self.tip = None #start with no text showing.
                
                self.widget.bind("<Enter>", self.show_tip) #show text when mouse enters widget.
                self.widget.bind("<Leave>", self.hide_tip) #hide text when mouse leaves widget.
        
            def show_tip(self, event=None): #when mouse enters help widget.
                x = self.widget.winfo_rootx() + 20
                y = self.widget.winfo_rooty() + 20 #find coordinates of "?" icon, then move 20 right and up.

                self.tip = tw = tk.Toplevel(self.widget) #create a top level window for the popup.
                tw.wm_overrideredirect(True) #specifies this is a blank window.
                tw.wm_geometry(f"+{x}+{y}") #shift window to avoid overlapping with "?".

                label = tk.Label(
                    tw,
                    text=self.text,
                    justify="left",
                    background="#ffffe0",
                    relief="solid",
                    borderwidth=1,
                    padx=6,
                    pady=4
                ) #customise appearance of label inside window.
                label.pack() #place label inside tooltip window.

            def hide_tip(self, event=None):
                    if self.tip is not None:
                        self.tip.destroy() #remove window.
                        self.tip = None #...and the reference to it.


    def populate_sample_entries(samples):
        rebuild_sample_rows(len(samples))  # create enough visible rows for the CSV

        for lon_entry, lat_entry, height_entry, heading_entry in sample_entries:
            lon_entry.delete(0, tk.END)
            lat_entry.delete(0, tk.END)
            height_entry.delete(0, tk.END)
            heading_entry.delete(0, tk.END)  # clear old values first

        for i, sample in enumerate(samples):
            lon_entry, lat_entry, height_entry, heading_entry = sample_entries[i]

            lon_entry.insert(0, str(sample["lon"]))
            lat_entry.insert(0, str(sample["lat"]))
            height_entry.insert(0, str(sample["observer_height"]))
            heading_entry.insert(0, str(sample["heading_deg"]))

        reposition_bottom_widgets()  # move Submit/Error underneath the new row count
        update_left_scrollregion()

    def load_metadata_csv(file_path):
        samples = [] #will store sample data.

        with open(file_path, newline="", encoding="utf-8-sig") as csvfile: #open CSV.
            reader = csv.DictReader(csvfile) #read in data as a dictionary.

            required_columns = {"lon", "lat", "observer_height", "heading_deg"} #assert which columns are needed.
            if reader.fieldnames is None:
                raise ValueError("The CSV file has no header row.") #if there is no header row, raise an error.

            missing = required_columns - set(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV file is missing required columns: {', '.join(sorted(missing))}") #if there are any missing columns, raise an error.

            for row_index, row in enumerate(reader, start=2): #start at 2 because row 1 is the header.
                try:
                    lon = float(row["lon"])
                    lat = float(row["lat"])
                    observer_height = float(row["observer_height"]) #extract longitude, latitutde and observer height data.
                    heading_deg = float(row["heading_deg"]) #extract heading data.
                except ValueError:
                    raise ValueError(f"Invalid numeric value in CSV row {row_index}.") #if any data is not a number, raise an error.

                if not (-180 <= lon <= 180):
                    raise ValueError(f"Longitude out of range in CSV row {row_index}.") #if any longitude is out of range, raise an error.

                if not (-90 <= lat <= 90):
                    raise ValueError(f"Latitude out of range in CSV row {row_index}.") #if any latitude is out of range, raise an error.

                if observer_height <= 0:
                    raise ValueError(f"Observer height must be greater than 0 in CSV row {row_index}.") #if height is 0 or less, raise an error.

                if not (0 <= heading_deg < 360):
                    raise ValueError(f"Heading must be between 0 and 360 degrees in CSV row {row_index}.") #if heading is out of range, raise an error.

                samples.append({
                    "lon": lon,
                    "lat": lat,
                    "observer_height": observer_height,
                    "heading_deg": heading_deg
                }) #add data to samples.

        return samples
    
    def metadata_handler(): 
        selected_metadata_var = tk.StringVar(value=metadata_csv_path if metadata_csv_path else "No metadata CSV selected") #Tkinter text displaying the current CSV path or "No metadata CSV selected".

        def load_metadata_file(): 
            global metadata_csv_path 
            global loaded_sample_metadata 

            file_path = filedialog.askopenfilename( #open a file picker window and store the chosen file path in file_path.
                parent=root, #pop up must be shown in the app.
                title="Open metadata CSV for VISTA", #text at the top of window.
                initialdir=".", #begin browsing in the directory of the program.
                filetypes=[
                    ("CSV files", "*.csv"), #only show files ending in .csv as the type we want the user to pick.
                ] #files that are shown as acceptable, in this case CSV.
            ) #use file explorer to allow the user to select a file.

            if not file_path: #check whether the user cancelled the file picker instead of selecting a file.
                return  #if user aborts, stop function.

            try:
                samples = load_metadata_csv(file_path) #read the CSV file and turn it into a validated list of sample dictionaries.
                metadata_csv_path = file_path #store the chosen CSV path.
                loaded_sample_metadata = samples #store the loaded sample list.
                selected_metadata_var.set(file_path) #update the Tkinter text variable to display the chosen CSV file path.
                populate_sample_entries(samples) #fill the visible entry boxes in the GUI with the values loaded from the CSV file.
                error_label.config(text="") #clear any old error message because loading succeeded.
            except Exception as e: 
                messagebox.showerror("Load error", f"Could not load metadata CSV:\n{e}") #if there is an error with loading the file, alert the user.

        return selected_metadata_var, load_metadata_file 
   
   
   
    def validate_inputs():
        max_observer_height = 10000
        samples = []

        for idx, (lon_entry, lat_entry, height_entry, heading_entry) in enumerate(sample_entries, start=1):
            lon_text = lon_entry.get().strip()
            lat_text = lat_entry.get().strip()
            observer_height_text = height_entry.get().strip()
            heading_text = heading_entry.get().strip()

            # if the whole row is blank, just ignore it
            if not any([lon_text, lat_text, observer_height_text, heading_text]):
                continue

            # but if the row is only partly filled, that is an error
            if not all([lon_text, lat_text, observer_height_text, heading_text]):
                raise ValueError(
                    f"Please fill in longitude, latitude, observer height, and heading for sample {idx}."
                )

            try:
                lon = float(lon_text)
                lat = float(lat_text)
                observer_height = float(observer_height_text)
                heading_deg = float(heading_text)
            except ValueError:
                raise ValueError(
                    f"Longitude, latitude, observer height, and heading must all be numbers for sample {idx}."
                )

            if not (-180 <= lon <= 180):
                raise ValueError(f"Longitude must be between -180 and 180 for sample {idx}.")

            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude must be between -90 and 90 for sample {idx}.")

            if observer_height <= 0:
                raise ValueError(f"Observer height must be greater than 0 metres for sample {idx}.")

            if observer_height > max_observer_height:
                raise ValueError(
                    f"Observer height must not exceed {max_observer_height} metres for sample {idx}."
                )

            if not (0 <= heading_deg < 360):
                raise ValueError(f"Heading must be between 0 and 360 degrees for sample {idx}.")

            samples.append({
                "lon": lon,
                "lat": lat,
                "observer_height": observer_height,
                "heading_deg": heading_deg
            })

        if not samples:
            raise ValueError("Please enter at least one sample or load a metadata CSV.")

        return samples
    
    def validate_max_distance():
        value_text = max_distance_var.get().strip()
        if not value_text:
            raise ValueError("Please enter a maximum distance.")
        try:
            value = float(value_text)
        except ValueError:
            raise ValueError("Maximum distance must be a number.")
        if value <= 0:
            raise ValueError("Maximum distance must be greater than 0 metres.")
        return value
        
    def show_error(message):
            error_label.config(text=message) #update the error label with the error text.

    def file_handler():
        
            selected_file_var = tk.StringVar(value=tif_path if tif_path else "No file selected") #default display when no file selected.

            def validate_file(file_path):
                ext = Path(file_path).suffix.lower() #obtain ending of file name.

                if ext not in [".tif", ".tiff"]: #if ending isn't ".tif" or ".tiff"...
                    raise ValueError(f"Unsupported file type: {ext}") #...raise an error.
                    
            def load_file():
                global tif_path
                file_path = filedialog.askopenfilename(
                    parent=root, #pop up must be shown in the app.
                    title="Open file for VISTA", #text at the top of window.
                    initialdir=".", #begin browsing in the directory of the program.
                    filetypes=[
                        ("GeoTIFF files", "*.tif *.tiff"),
                    ] #files that are shown as acceptable, in this case TIFF.
                ) #use file explorer to allow the user to select a file.

                if not file_path:
                    return  #if user aborts, stop function.

                try:
                    validate_file(file_path) #validate file function.
                    tif_path = file_path #store chosen file path to be passed into run_program later.
                    selected_file_var.set(file_path)
                    right_sidebar.load_dem(tif_path) #load initial DEM preview.
                    error_label.config(text="")
                except Exception as e:
                    messagebox.showerror("Load error", f"Could not load file:\n{e}") #if there is an error with loading the file, alert the user.

            top_bar = ttk.Frame(root, padding=8) #create a container for the file button and text.
            top_bar.grid(row=0, column=0, columnspan=2, sticky="ew") #stretch horizontally.

            file_button = ttk.Button(top_bar, text="DEM", command=load_file)
            file_button.pack(side="left") #move to the leftmost side.

            file_label = ttk.Label(top_bar, textvariable=selected_file_var) #label for currently selected file, or default text if no file loaded.
            file_label.pack(side="left", padx=10) #move to the leftmost side, next to the button.

            metadata_button = ttk.Button(top_bar, text="CSV", command=load_metadata_file) #create button that runs "load_metadata_file" when clicked.
            metadata_button.pack(side="left", padx=(20, 0)) #move to the leftmost side, next to the file path.

            metadata_label = ttk.Label(top_bar, textvariable=selected_metadata_var) #label for currently selected metadata CSV, or default text if no file loaded.
            metadata_label.pack(side="left", padx=10) #move to the leftmost side, next to the button.

            max_distance_label = ttk.Label(top_bar, text="Max distance (m):")
            max_distance_label.pack(side="left", padx=(20, 4))

            max_distance_entry = ttk.Entry(top_bar, width=10, textvariable=max_distance_var)
            max_distance_entry.pack(side="left")

            
    def submit():
            global tif_path
            error_label.config(text="") # clear any previous error message.
            try:
                if not tif_path:
                    raise ValueError("Please select a GeoTIFF file first.")

                sample_metadata = validate_inputs() #validate the user's inputs.
                max_distance = validate_max_distance()

                right_sidebar.load_dem(tif_path) #load DEM into preview.

                result = run_program(
                    sample_metadata,
                    tif_path,
                    max_distance,
                ) #run the main program with the three values on the embedded axes.

                right_sidebar.set_results(
                    result["count_overlay"],
                    observer_points=result["observer_points_xy"],
                    view_extent=result["view_extent"],
                    scale_bar_length_m=result["scale_bar_length_m"]
                )


                right_sidebar.canvas.draw_idle() #refresh the embedded preview.

            except ValueError as e:
                show_error(str(e)) #handle invalid number input.
            except Exception as e:
                show_error(str(e)) #handle generic bad user input.
   
    global tif_path 
    root = tk.Tk() #create the GUI window.
    root.title("VISTA") #title the window.
    root.geometry("1350x760") #default window size.
    root.resizable(True, True) #allow user to resize window.

    root.rowconfigure(1, weight=1)
    root.columnconfigure(1, weight=1) #define region for file bar.

    left_container = ttk.Frame(root) #outer container for scrollable left panel.
    left_container.grid(row=1, column=0, sticky="nsew") #place container in left side of main window.

    left_container.rowconfigure(0, weight=1)
    left_container.columnconfigure(0, weight=1)

    left_canvas = tk.Canvas(left_container, highlightthickness=0, width=620) #canvas that will scroll.
    left_canvas.grid(row=0, column=0, sticky="nsew")

    left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview) #vertical scrollbar.
    left_scrollbar.grid(row=0, column=1, sticky="ns")

    left_canvas.configure(yscrollcommand=left_scrollbar.set)

    left_panel = ttk.Frame(left_canvas, padding=12) #actual content frame that holds all widgets.
    left_panel_window = left_canvas.create_window((0, 0), window=left_panel, anchor="nw") #place frame inside canvas.

    def update_left_scrollregion(event=None):
        left_canvas.configure(scrollregion=left_canvas.bbox("all")) #tell canvas how tall the scrollable area is.

    def resize_left_panel_width(event):
        left_canvas.itemconfigure(left_panel_window, width=event.width) #keep inner frame same width as canvas.

    left_panel.bind("<Configure>", update_left_scrollregion)
    left_canvas.bind("<Configure>", resize_left_panel_width)

    def on_mousewheel_windows(event):
        left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units") #mouse wheel scroll on Windows.

    def on_mousewheel_linux_up(event):
        left_canvas.yview_scroll(-1, "units") #Linux scroll up.

    def on_mousewheel_linux_down(event):
        left_canvas.yview_scroll(1, "units") #Linux scroll down.

    def bind_mousewheel(event=None):
        left_canvas.bind_all("<MouseWheel>", on_mousewheel_windows)
        left_canvas.bind_all("<Button-4>", on_mousewheel_linux_up)
        left_canvas.bind_all("<Button-5>", on_mousewheel_linux_down)

    def unbind_mousewheel(event=None):
        left_canvas.unbind_all("<MouseWheel>")
        left_canvas.unbind_all("<Button-4>")
        left_canvas.unbind_all("<Button-5>")

    left_canvas.bind("<Enter>", bind_mousewheel)
    left_canvas.bind("<Leave>", unbind_mousewheel)

    max_distance_var = tk.StringVar(value="500.0")

    selected_metadata_var, load_metadata_file = metadata_handler() #create button to load in a CSV by defining variable and function with does so.
    right_sidebar = RightSideBar(root) #create right hand pannel.
    right_sidebar.grid(row=1, column=1, sticky="nsew") #define region for right panel.

    selected_row_var = tk.IntVar(value=1)
    sample_entries = []

    tk.Label(left_panel, text="Sample").grid(row=1, column=0, padx=(12, 4), pady=8, sticky="w") #add table heading.
    tk.Label(left_panel, text="Longitude (EPSG:4326):").grid(row=1, column=1, padx=(12, 4), pady=8, sticky="w") #add text for longitude input box, push it to the left and add padding.
    lon_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow") #create the helper widget and make the foreground blue.
    lon_help.grid(row=1, column=2, padx=6, pady=8, sticky="w") #place helper widget into grid.

    tk.Label(left_panel, text="Latitude (EPSG:4326):").grid(row=1, column=3, padx=(12, 4), pady=8, sticky="w") #add text for latitude input box, push it to the left and add padding.
    lat_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow") #create the helper widget and make the foreground blue.
    lat_help.grid(row=1, column=4, padx=6, pady=8, sticky="w") #place helper widget into grid.

    tk.Label(left_panel, text="Observer height (m):").grid(row=1, column=5, padx=(12, 4), pady=8, sticky="w") #add text for observer height input box, push it to the left and add padding.
    height_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow") #create the helper widget and make the foreground blue.
    height_help.grid(row=1, column=6, padx=6, pady=8, sticky="w") #place helper widget into grid.

    tk.Label(left_panel, text="Heading (deg):").grid(row=1, column=7, padx=(12, 4), pady=8, sticky="w") #add text for heading input box, push it to the left and add padding.
    heading_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow") #create the helper widget and make the foreground blue.
    heading_help.grid(row=1, column=8, padx=6, pady=8, sticky="w") #place helper widget into grid.

    sample_entries = []
    sample_row_widgets = []

    def create_sample_row(i):
        grid_row = i + 2  # row 1 is the heading row

        row_label = tk.Label(left_panel, text=f"{i + 1}.")
        row_label.grid(row=grid_row, column=0, padx=(12, 4), pady=4, sticky="w")

        lon_entry = tk.Entry(left_panel, width=16)
        lon_entry.grid(row=grid_row, column=1, pady=4, sticky="w")

        lat_entry = tk.Entry(left_panel, width=16)
        lat_entry.grid(row=grid_row, column=3, pady=4, sticky="w")

        height_entry = tk.Entry(left_panel, width=12)
        height_entry.grid(row=grid_row, column=5, pady=4, sticky="w")

        heading_entry = tk.Entry(left_panel, width=12)
        heading_entry.grid(row=grid_row, column=7, pady=4, sticky="w")

        sample_entries.append((lon_entry, lat_entry, height_entry, heading_entry))
        sample_row_widgets.append((row_label, lon_entry, lat_entry, height_entry, heading_entry))

    def rebuild_sample_rows(count):
        count = max(1, count)  # keep at least 1 visible row for manual entry.

        for widgets in sample_row_widgets:
            for widget in widgets:
                widget.destroy()

        sample_entries.clear()
        sample_row_widgets.clear()

        for i in range(count):
            create_sample_row(i)

    def add_blank_row():
        create_sample_row(len(sample_entries))
        reposition_bottom_widgets()
        update_left_scrollregion()

    def delete_last_row():
        if len(sample_row_widgets) <= 1:
            error_label.config(text="At least one sample row must remain.")
            return
        widgets = sample_row_widgets.pop() #take the final row.
        for widget in widgets:
            widget.destroy() #remove its widgets from the GUI.

        sample_entries.pop() #remove matching entries tuple.
        error_label.config(text="") #clear any old error message.
        reposition_bottom_widgets()
        update_left_scrollregion()

    def reposition_bottom_widgets():
        submit_row = len(sample_entries) + 2
        actions_row = len(sample_entries) + 3
        error_row = len(sample_entries) + 4

        submit_button.grid_configure(row=submit_row)
        add_row_button.grid_configure(row=actions_row)
        delete_row_button.grid_configure(row=actions_row)
        error_label.grid_configure(row=error_row)


    right_sidebar.point_selected_callback = set_coordinate_entries # assign function to update coordinates.

    rebuild_sample_rows(1)  # start with 1 blank row

    submit_button = tk.Button(left_panel, text="Submit", command=submit) #run the program.
    submit_button.grid(row=999, column=0, columnspan=9, pady=(18, 10)) #temporary row, corrected below.

    add_row_button = tk.Button(left_panel, text="Add sample", command=add_blank_row) #append one blank row.
    add_row_button.grid(row=999, column=0, columnspan=2, pady=(0, 10), sticky="w")

    delete_row_button = tk.Button(left_panel, text="Delete last row", command=delete_last_row) #remove final row.
    delete_row_button.grid(row=999, column=2, columnspan=2, pady=(0, 10), sticky="e")

    error_label = tk.Label(left_panel, text="", fg="red")
    error_label.grid(row=999, column=0, columnspan=9, pady=(0, 10))

    reposition_bottom_widgets()
    update_left_scrollregion()
    reposition_bottom_widgets()

    file_handler()

    #attach a tooltip to the latitude help widget.
    LeftSideBar(
        lon_help,
        "Enter the coordinate's longitude in EPSG:4326.\nExample: -1.3276"
    )

    #attach a tooltip to the latitude help widget.
    LeftSideBar(
        lat_help,
        "Enter the coordinate's latitude in EPSG:4326.\nExample: 50.730251"
    )

    #attach a tooltip to the observer height help widget.
    LeftSideBar(
        height_help,
        "Enter observer height above the ground in metres.\nExample: 1.5"
    )

    #attach a tooltip to the heading help widget.
    LeftSideBar(
        heading_help,
        "Enter the viewing direction as a compass bearing in degrees.\n0 = north, 90 = east, 180 = south, 270 = west"
    )

    if tif_path:
        right_sidebar.load_dem(tif_path) #load initial DEM preview.

    #start Tkinter event loop so it "listens" for user input.
    root.mainloop()