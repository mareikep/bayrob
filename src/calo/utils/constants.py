###########
# CLASSES #
###########
import csv
from _csv import QUOTE_MINIMAL, register_dialect
import xlwt

############
# PRINTING #
############
cs = ',\n'
nl = '\n'
cst = '\n\t'

###########
# PROJECT #
###########
def projectnameUP(): 'CALO'
def projectnameLOW(): 'calo'
APPNAME = projectnameUP.__doc__
APPAUTHOR = 'picklum'

###########
# LOGGING #
###########
calologger = f'/calo'
calojsonlogger = f'/calo/json'
calofileloggerr = f'/calo/file/res'
calofileloggerv = f'/calo/file/verbose'
connectionlogger = f'/calo/connection'

resultlog = f'{projectnameUP.__doc__}_res-{{}}.log'
logs = f'{projectnameUP.__doc__}_log-{{}}.log'

#################
# TIME AND DATE #
#################
TMPFILESTRFMT = 'TMP_%Y%m%d_%H-%M-%S'
FILESTRFMT = "%Y-%m-%d_%H:%M"
FILESTRFMT_NOTIME = "%Y-%m-%d"
FILESTRFMT_SEC = "%Y-%m-%d_%H:%M:%S"


#######
# CSV #
#######
class CALODialect(csv.Dialect):
    """Describe the properties of Unix-generated CSV files."""
    delimiter = ';'
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = '\n'
    quoting = QUOTE_MINIMAL


register_dialect("calodialect", CALODialect)

###############
# TEXT STYLES #
###############
xlsHEADER = xlwt.easyxf('font: name monospace, color-index black, bold on', num_format_str='#,##0.00000000000000000000')
xlsDATE = xlwt.easyxf(num_format_str='D-MMM-YY')
xlsNUM = xlwt.easyxf(num_format_str='#,##0.00000000000000000000')

##########
# COLORS #
##########
lightblue = '#b3ffff80'
darkblue = '#3f9fff80'
green = '#b3ff4c80'
yellow = '#ffff4c80'
orange = '#ffbe4980'
red = '#FF4A4980'

plotcolormap = 'tab20b'  # or viridis
plotstyle = 'seaborn-deep'
avalailable_colormaps = [
    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu',
    'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r',
    'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired',
    'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu',
    'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r',
    'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn',
    'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
    'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r',
    'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r',
    'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis',
    'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix',
    'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r',
    'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r',
    'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2',
    'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno',
    'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean',
    'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
    'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10',
    'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain',
    'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted',
    'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
]

##########
# ERRORS #
##########

TABLEFORMAT = """<br><br><b>The data you uploaded does not have the right format. You can analyze your data if it is a semicolon-separated .csv file in the following format:</b><br><br>
<ul>
  <li>The first row contains the header information. Feature names that start with "target_" will automatically be considered target features.<br>This may not make a difference for some of the analysis tools. Unit information for the respective column can be added by adding it brackets, i.e. "<featurename> [<unit>], e.g. %, MPa, °C. Leave empty if there is no unit.</li>
  <li>Each of the following rows is considered one training example. Values can be numeric, i.e. integers or floats or symbolic, i.e. categorical or boolean. <br><b>Exception 1:</b> the 'id' value can be of type string. <br> </li>
  <li>Missing values can be identified by inserting a default value that can be replaced later.</li>
  <li>(optional) If a column named 'id' exists, its values will serve as identifiers for each training sample, which allows to investigate inference results later and retrieve examples the results base on.<br>If this column does not exists or its values are empty, each sample will automatically be assigned an id in ascending order.</li>
</ul>  
<br>
<b>Example:</b>
<br>
<table>
  <tr>
    <th style="border: 1px solid #dddddd;text-align: center;">id</th>
    <th style="border: 1px solid #dddddd;text-align: center;">num_passengers</th>
    <th style="border: 1px solid #dddddd;text-align: center;">...</th>
    <th style="border: 1px solid #dddddd;text-align: center;">avg_speed</th>
    <th style="border: 1px solid #dddddd;text-align: center;">target_distance</th>
    <th style="border: 1px solid #dddddd;text-align: center;">...</th>
    <th style="border: 1px solid #dddddd;text-align: center;">target_fuel_savings</th>
  </tr>
  <tr>
    <td style="border: 1px solid #dddddd;text-align: center;"></td>
    <td style="border: 1px solid #dddddd;text-align: center;"></td>
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
    <td style="border: 1px solid #dddddd;text-align: center;">km/h</td>
    <td style="border: 1px solid #dddddd;text-align: center;">km</td>
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
    <td style="border: 1px solid #dddddd;text-align: center;">%</td>
  </tr>
  <tr style="background-color: #dddddd;">
    <td style="border: 1px solid #dddddd;text-align: center;">e_0</td>
    <td style="border: 1px solid #dddddd;text-align: center;">1</td>
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
    <td style="border: 1px solid #dddddd;text-align: center;">75</td>
    <td style="border: 1px solid #dddddd;text-align: center;">400.4</td>
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
    <td style="border: 1px solid #dddddd;text-align: center;">0.1234</td>
  </tr>  
  <tr>
    <td style="border: 1px solid #dddddd;text-align: center;">e_1</td>
    <td style="border: 1px solid #dddddd;text-align: center;">3</td>
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
    <td style="border: 1px solid #dddddd;text-align: center;">61</td>
    <td style="border: 1px solid #dddddd;text-align: center;">1000.3</td>
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
    <td style="border: 1px solid #dddddd;text-align: center;">0.98</td>
  </tr>
  <tr style="background-color: #dddddd;">
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
  </tr>  
  <tr>
    <td style="border: 1px solid #dddddd;text-align: center;">e_t</td>
    <td style="border: 1px solid #dddddd;text-align: center;">4</td>
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
    <td style="border: 1px solid #dddddd;text-align: center;">100</td>
    <td style="border: 1px solid #dddddd;text-align: center;">810.7</td>
    <td style="border: 1px solid #dddddd;text-align: center;">...</td>
    <td style="border: 1px solid #dddddd;text-align: center;">0.56</td>
  </tr>
</table>

"""
