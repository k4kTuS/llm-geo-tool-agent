import xml.etree.ElementTree as ET
import re
import requests

from config.wms import wms_config

def parse_sld(endpoint: str, layer: str) -> tuple[dict, dict] | None:
    try:
        api_setup = wms_config[endpoint]
        response = requests.get(
            api_setup["wms_root_url"],
            params={
                "map": api_setup["data"]["map"],
                "version": "1.3.0",
                "service": "WMS",
                "request": "GetStyles",
                "layers": layer,
            },
            stream=True
        )
        xml_content = response.content.decode('utf-8')
        layer_rules = parse_rules(xml_content)
        rules = list(layer_rules.values())[0] # Use only first layer
        class_to_rgb, rgb_to_class = build_color_mapping(rules)
        return class_to_rgb, rgb_to_class
    except Exception as e:
        print("SDL Parsing error:", e)
        return None, None

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def parse_rules(xml_content: str) -> dict:
    tree = ET.ElementTree(ET.fromstring(xml_content))
    root = tree.getroot()

    namespaces = {
        'sld': 'http://www.opengis.net/sld',
        'ogc': 'http://www.opengis.net/ogc',
        'se': 'http://www.opengis.net/se'
    }

    styles_dict = {}
    for named_layer in root.findall('.//sld:NamedLayer', namespaces):
        layer_name = named_layer.find('se:Name', namespaces)
        layer_name = layer_name.text if layer_name is not None else 'Unnamed Layer'
        styles_dict[layer_name] = {}

        for style in named_layer.findall('.//sld:UserStyle', namespaces):
            for rule in style.findall('.//se:Rule', namespaces):
                rule_name_el = rule.find('se:Name', namespaces)
                if rule_name_el is None:
                    continue
                rule_name = rule_name_el.text

                clc_id_elem = rule.find('.//ogc:Literal', namespaces)
                if clc_id_elem is None:
                    continue
                clc_id = clc_id_elem.text

                polygon_symbolizer = rule.find('.//se:PolygonSymbolizer', namespaces)
                fill_color = None
                if polygon_symbolizer is not None:
                    fill_el = polygon_symbolizer.find('.//se:Fill/se:SvgParameter[@name="fill"]', namespaces)
                    if fill_el is not None:
                        fill_color = fill_el.text
                if fill_color is None:
                    continue
                styles_dict[layer_name][clc_id] = {
                    'rule': rule_name,
                    'fill_color': fill_color
                }

    return styles_dict

def build_color_mapping(data: dict):
    class_levels = {}

    for clc_id, rule_data in sorted(data.items()):
        level = len(clc_id.split("0")[0])
        fill_color = rule_data["fill_color"]
        name = rule_data["rule"]

        if class_levels.get(fill_color) is None:
            class_levels[fill_color] = {}
        if class_levels[fill_color].get(level) is None:
            class_levels[fill_color][level] = {
                "ids": [clc_id],
                "names": [name]
            }
        else:
            class_levels[fill_color][level]["ids"].append(clc_id)
            class_levels[fill_color][level]["names"].append(name)
    color_mapping = {}
    for clr, levels in class_levels.items():
        color_mapping[clr] = {}
        # One color per class group - Use class that covers all sublcasses
        lvl_keys = sorted(levels.keys(), reverse=True)
        for lvl in lvl_keys:
            if len(levels[lvl]["ids"]) == 1:
                color_mapping[clr]["name"] = levels[lvl]["names"][0]
                color_mapping[clr]["id"] = levels[lvl]["ids"][0]
                break
        # Multiple classes from different groups have the same color - use the last parsed rule
        color_mapping[clr]["name"] = levels[lvl_keys[0]]["names"][-1]
        color_mapping[clr]["id"] = levels[lvl_keys[0]]["ids"][-1]
    for k, v in color_mapping.items():
        name_cleared = re.sub(r'([a-z])([A-Z])', r'\1 \2', v["name"].split("_")[-1])
        color_mapping[k]["name"] = name_cleared
        
    layer_to_rgb = {}
    rgb_to_layer = {}
    for clr, rule_data in color_mapping.items():
        layer_to_rgb[rule_data["name"]] = hex_to_rgb(clr)
        rgb_to_layer[hex_to_rgb(clr)] = rule_data["name"]
    return layer_to_rgb, rgb_to_layer