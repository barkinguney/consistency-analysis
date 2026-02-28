"""Disclaimer: Created this script mostly with LLMs. Might not cover all cases."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import xml.etree.ElementTree as ET
import cantera as ct

import cantera_related_functions  # for unit conversion


Number = Union[int, float]


def _strip_ns(tag: str) -> str:
    """Remove XML namespace if present: {ns}tag -> tag."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _as_number(s: Optional[str]) -> Optional[Number]:
    """Parse int/float from text; return None on missing/empty."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # try int first for cleaner values
    try:
        i = int(s)
        return i
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return None


def _child_text(el: ET.Element, child_tag: str) -> Optional[str]:
    """Get direct child text (ignores namespaces)."""
    for ch in list(el):
        if _strip_ns(ch.tag) == child_tag:
            return ch.text
    return None


def _find_first(root: ET.Element, tag: str) -> Optional[ET.Element]:
    """Find first element by localname tag anywhere under root."""
    for el in root.iter():
        if _strip_ns(el.tag) == tag:
            return el
    return None


def _find_all(parent: ET.Element, tag: str) -> List[ET.Element]:
    """Find all elements by localname tag anywhere under parent."""
    out: List[ET.Element] = []
    for el in parent.iter():
        if _strip_ns(el.tag) == tag:
            out.append(el)
    return out


@dataclass
class IDTPropertyDef:
    prop_id: str
    name: str
    label: Optional[str]
    units: Optional[str]
    sourcetype: Optional[str]


@dataclass
class IDTDataGroup:
    group_id: str
    # property definitions keyed by x-id (e.g., x1, x2, x3)
    properties: Dict[str, IDTPropertyDef]
    # rows as dicts keyed by x-id AND also by label/name for convenience
    rows: List[Dict[str, Any]]


@dataclass
class IDTExperiment:
    file_author: Optional[str]
    file_doi: Optional[str]
    experiment_type: Optional[str]
    apparatus_kind: Optional[str]
    apparatus_mode: Optional[str]
    initial_composition: List[Dict[str, Any]]
    data_groups: List[IDTDataGroup]
    ignition_type: Optional[IgnitionType]
    common_properties: Dict[str, Any]
    common_property_units: Dict[str, Optional[str]]
    bibliography_author: Optional[str]
    bibliography_year: Optional[str]
    
@dataclass
class IgnitionType:
    target: Optional[str]
    type: Optional[str]
    amount: Optional[str]
    units: Optional[str]


def _author_before_first_comma(author: Optional[str]) -> str:
    """Return author text truncated before the first comma."""
    if not author:
        return ""
    author = author.strip()
    for idx, ch in enumerate(author):
        if ch == ",":
            return author[:idx].strip()
    return author


def _clean_author_letters_only(author: str) -> str:
    """Keep only letters from author text and remove whitespace."""
    return "".join(ch for ch in author if ch.isalpha())


def parse_idt_xml(path: Union[str, Path]) -> IDTExperiment:
    """
    Parse an IDT (ignition delay time) XML file in the shown ReSpecTh-like format.

    Returns a structured IDTExperiment containing metadata, composition, and all dataGroups.
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    # If the file uses namespaces, ElementTree tags will be {ns}experiment etc.
    # We search by localname to stay robust.

    exp = root if _strip_ns(root.tag) == "experiment" else _find_first(root, "experiment")
    if exp is None:
        raise ValueError("Could not find <experiment> root element")

    file_author = _child_text(exp, "fileAuthor")
    file_doi = _child_text(exp, "fileDOI")
    experiment_type = _child_text(exp, "experimentType")

    bibliography_author = None
    bibliography_year = None
    bib_link = _find_first(exp, "bibliographyLink")
    if bib_link is not None:
        details_el = _find_first(bib_link, "details")
        if details_el is not None:
            bibliography_author = _child_text(details_el, "author")
            bibliography_year = _child_text(details_el, "year")

    apparatus = _find_first(exp, "apparatus")
    apparatus_kind = _child_text(apparatus, "kind") if apparatus is not None else None
    apparatus_mode = _child_text(apparatus, "mode") if apparatus is not None else None

    # Parse initial composition (commonProperties/property[@name="initial composition"]/component...)
    initial_composition: List[Dict[str, Any]] = []
    common_props = _find_first(exp, "commonProperties")
    if common_props is not None:
        for prop in list(common_props):
            if _strip_ns(prop.tag) != "property":
                continue
            if prop.attrib.get("name", "").strip().lower() != "initial composition":
                continue

            for comp in list(prop):
                if _strip_ns(comp.tag) != "component":
                    continue

                species_link = None
                amount_el = None
                for ch in list(comp):
                    t = _strip_ns(ch.tag)
                    if t == "speciesLink":
                        species_link = ch
                    elif t == "amount":
                        amount_el = ch

                species: Dict[str, Any] = {}
                if species_link is not None:
                    species = {
                        "preferredKey": species_link.attrib.get("preferredKey"),
                        "chemName": species_link.attrib.get("chemName"),
                        "CAS": species_link.attrib.get("CAS"),
                        "InChI": species_link.attrib.get("InChI"),
                        "SMILES": species_link.attrib.get("SMILES"),
                    }

                amount: Dict[str, Any] = {}
                if amount_el is not None:
                    amount = {
                        "units": amount_el.attrib.get("units"),
                        "value": _as_number(amount_el.text),
                    }

                if species or amount:
                    initial_composition.append({**species, **amount})

            break  # only one initial composition property expected
    
    # Parse ignition type
    ign_el = _find_first(exp, "ignitionType")
    ignition_type = None
    if ign_el is not None:
        ignition_type = IgnitionType(
            target=ign_el.attrib.get("target"),
            type=ign_el.attrib.get("type"),
            amount=ign_el.attrib.get("amount"),  # Ensure amount is included
            units=ign_el.attrib.get("units")     # Ensure units is included
        )

    # Parse data groups
    data_groups: List[IDTDataGroup] = []
    for dg in _find_all(exp, "dataGroup"):
        group_id = dg.attrib.get("id", "")

        # property definitions are direct children <property .../>
        props: Dict[str, IDTPropertyDef] = {}
        for ch in list(dg):
            if _strip_ns(ch.tag) != "property":
                continue
            pid = ch.attrib.get("id")
            if not pid:
                continue
            props[pid] = IDTPropertyDef(
                prop_id=pid,
                name=ch.attrib.get("name", ""),
                label=ch.attrib.get("label"),
                units=ch.attrib.get("units"),
                sourcetype=ch.attrib.get("sourcetype"),
            )

        # datapoints: each <dataPoint><x1>..</x1><x2>..</x2>..</dataPoint>
        rows: List[Dict[str, Any]] = []
        for dp in list(dg):
            if _strip_ns(dp.tag) != "dataPoint":
                continue

            row: Dict[str, Any] = {}
            for val_el in list(dp):
                xid = _strip_ns(val_el.tag)  # x1, x2, x3...
                row[xid] = _as_number(val_el.text)

                # also add convenience keys by label and by name if available
                if xid in props:
                    pdef = props[xid]
                    if pdef.label:
                        row[pdef.label] = row[xid]
                    if pdef.name:
                        row[pdef.name] = row[xid]

            rows.append(row)

        data_groups.append(IDTDataGroup(group_id=group_id, properties=props, rows=rows))

    # Parse common properties (temperature, pressure outside dataGroups)
    common_properties: Dict[str, Any] = {}
    common_property_units: Dict[str, Optional[str]] = {}
    common_props = _find_first(exp, "commonProperties")
    if common_props is not None:
        for prop in list(common_props):
            if _strip_ns(prop.tag) != "property":
                continue
            prop_name = prop.attrib.get("name", "").strip().lower()
            if prop_name in ["temperature", "pressure"]:
                value_el = _find_first(prop, "value")
                if value_el is not None and value_el.text:
                    common_properties[prop_name] = _as_number(value_el.text)
                    common_property_units[prop_name] = prop.attrib.get("units") or value_el.attrib.get("units") if hasattr(value_el, "attrib") else None
    
    return IDTExperiment(
        file_author=file_author,
        file_doi=file_doi,
        experiment_type=experiment_type,
        apparatus_kind=apparatus_kind,
        apparatus_mode=apparatus_mode,
        initial_composition=initial_composition,
        data_groups=data_groups,
        ignition_type=ignition_type,
        common_properties=common_properties,
        common_property_units=common_property_units,
        bibliography_author=bibliography_author,
        bibliography_year=bibliography_year,
    )


def idt_to_dataframe(exp: IDTExperiment, group_id: Optional[str] = None):
    """
    Optional helper: convert a dataGroup to a pandas DataFrame.
    Requires pandas installed.
    """
    import pandas as pd  # type: ignore

    groups = exp.data_groups
    if group_id is not None:
        groups = [g for g in groups if g.group_id == group_id]
        if not groups:
            raise ValueError(f"No dataGroup found with id={group_id!r}")

    # Use first group by default
    g = groups[0]
    df = pd.DataFrame(g.rows)

    # Prefer columns in x-id order + (label) if present
    xids = sorted(g.properties.keys(), key=lambda s: int(s[1:]) if s[1:].isdigit() else s)
    preferred_cols: List[str] = []
    for xid in xids:
        preferred_cols.append(xid)
        lab = g.properties[xid].label
        if lab and lab in df.columns:
            preferred_cols.append(lab)

    # keep preferred first, then the rest
    remaining = [c for c in df.columns if c not in preferred_cols]
    df = df[preferred_cols + remaining]
    return df


def format_composition(initial_composition: List[Dict[str, Any]]) -> str:
    """
    Format initial composition as a string like 'C2H4:0.01, O2:0.03, AR:0.96'.
    Uses preferredKey as species identifier and value as the mole fraction.
    """
    parts = []
    for comp in initial_composition:
        species = comp.get("preferredKey", "")
        value = comp.get("value")
        if species and value is not None:
            parts.append(f"{species}:{value}")
    return ", ".join(parts)


def calculate_phi_cantera_old(
    composition: Union[str, List[Dict[str, Any]]],
    mechanism: str = "gri30.yaml",
    fuel: Optional[str] = None,  # kept for signature compatibility
    oxidizer: str = "O2:1.0",    # kept for signature compatibility
) -> Optional[float]:
    """
    Calculate equivalence ratio (phi) by balancing required O2 against available O2.
    Diluent species (e.g., Ar) do not affect the result.
    """
    try:
        gas = ct.Solution(mechanism)
        name_map = {sp.upper(): sp for sp in gas.species_names}

        # Parse composition to dict
        if isinstance(composition, str):
            comp_dict: Dict[str, float] = {}
            for part in composition.split(","):
                part = part.strip()
                if ":" in part:
                    sp, val = part.split(":")
                    try:
                        comp_dict[sp.strip()] = float(val.strip())
                    except ValueError:
                        continue
        else:
            comp_dict = {
                comp.get("preferredKey", ""): float(comp.get("value", 0))
                for comp in composition
                if comp.get("preferredKey") and comp.get("value") is not None
            }

        # Normalize species names to mechanism names
        norm_comp: Dict[str, float] = {}
        for sp, val in comp_dict.items():
            mech_sp = name_map.get(sp.upper())
            if mech_sp and val > 0:
                norm_comp[mech_sp] = val

        if not norm_comp:
            return None

        o2_available = norm_comp.get("O2", 0.0)
        if o2_available <= 0:
            return None

        # Total O2 needed to fully oxidize all fuel species
        total_o2_needed = 0.0
        for sp, moles in norm_comp.items():
            if sp == "O2" or moles <= 0:
                continue
            sp_obj = gas.species(sp)
            c = sp_obj.composition.get("C", 0.0)
            h = sp_obj.composition.get("H", 0.0)
            o = sp_obj.composition.get("O", 0.0)
            nu_o2 = c + h / 4.0 - o / 2.0
            if nu_o2 > 0:
                total_o2_needed += nu_o2 * moles

        if total_o2_needed <= 0:
            return None

        return total_o2_needed / o2_available
    except Exception as e:
        print(f"Error calculating phi with Cantera: {e}")
        return None

def calculate_phi(initial_composition):
    """
    Calculates the equivalence ratio (phi) for a given gas composition.
    
    Args:
        composition (dict): Species names as keys, mole fractions as values.
    """
    
    parts = {}
    for comp in initial_composition:
        species = comp.get("preferredKey", "")
        value = comp.get("value")
        if species and value is not None:
            parts[species] = value
    
    
    # 1. Define stoichiometric O2 coefficients (moles of O2 per mole of fuel)
    # Formula: s = C + H/4 - O/2
    stoich_coeffs = {
        'H2':  0.5,   # H2 + 0.5 O2 -> H2O
        'CO':  0.5,   # CO + 0.5 O2 -> CO2
        'CH4': 2.0,   # CH4 + 2 O2 -> CO2 + 2 H2O
        'C2H4': 3.0,  # C2H4 + 3 O2 -> 2 CO2 + 2 H2O
        'C2H6': 3.5,  # C2H6 + 3.5 O2 -> 2 CO2 + 3 H2O
        'C3H8': 5.0,  # C3H8 + 5 O2 -> 3 CO2 + 4 H2O
        'CH3OH': 1.5, # CH3OH + 1.5 O2 -> CO2 + 2 H2O
    }

    # 2. Identify actual O2 in the mixture
    actual_o2 = parts.get('O2', 0.0)
    
    if actual_o2 == 0:
        return float('inf')  # Pure fuel / No oxidizer

    # 3. Calculate total O2 required for stoichiometry
    required_o2 = 0.0
    for species, mole_fraction in parts.items():
        if species in stoich_coeffs:
            required_o2 += mole_fraction * stoich_coeffs[species]

    # 4. Calculate Phi
    # Phi = (Fuel/Oxidizer)_actual / (Fuel/Oxidizer)_stoich
    # Which simplifies to: Required_O2 / Actual_O2
    phi = required_o2 / actual_o2
    
    return phi


def add_phi_to_xml_copy(
    xml_path: Union[str, Path],
    output_folder: Union[str, Path],
) -> Path:
    """
    Read an IDT XML file, calculate phi from initial composition,
    convert pressure and ignition-delay values/units, and write a copy with
    commonProperties/property[name="phi"] added or updated.

    Args:
        xml_path: Input XML path.
        output_folder: Directory where the updated XML copy will be written.
        mechanism: Cantera mechanism used by phi calculation fallback.

    Returns:
        Path to the written XML copy.
    """
    xml_path = Path(xml_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    exp = parse_idt_xml(xml_path)
    phi_val = calculate_phi(exp.initial_composition)
    if phi_val is None:
        raise ValueError(f"Could not calculate phi for {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    exp_el = root if _strip_ns(root.tag) == "experiment" else _find_first(root, "experiment")
    if exp_el is None:
        raise ValueError("Could not find <experiment> root element")

    common_props = None
    for ch in list(exp_el):
        if _strip_ns(ch.tag) == "commonProperties":
            common_props = ch
            break
    if common_props is None:
        common_props = ET.SubElement(exp_el, "commonProperties")

    phi_prop = None
    for prop in list(common_props):
        if _strip_ns(prop.tag) != "property":
            continue
        if prop.attrib.get("name", "").strip().lower() == "phi":
            phi_prop = prop
            break
    if phi_prop is None:
        phi_prop = ET.SubElement(common_props, "property", {"name": "phi"})
    else:
        phi_prop.attrib["name"] = "phi"

    value_el = None
    for ch in list(phi_prop):
        if _strip_ns(ch.tag) == "value":
            value_el = ch
            break
    if value_el is None:
        value_el = ET.SubElement(phi_prop, "value")

    value_el.text = f"{float(phi_val):.12g}"

    # Convert pressure and ignition delay data in all dataGroups.
    for dg in _find_all(exp_el, "dataGroup"):
        pressure_pid = None
        tau_pid = None
        pressure_units_src = None
        tau_units_src = None
        pressure_units_out = None
        tau_units_out = None

        for ch in list(dg):
            if _strip_ns(ch.tag) != "property":
                continue

            pid = ch.attrib.get("id")
            if not pid:
                continue

            name = (ch.attrib.get("name") or "").strip().lower()
            label = (ch.attrib.get("label") or "").strip().lower()

            if pressure_pid is None and (name == "pressure" or label == "p"):
                pressure_pid = pid
                pressure_units_src = ch.attrib.get("units")
            if tau_pid is None and (name == "ignition delay" or label == "tau"):
                tau_pid = pid
                tau_units_src = ch.attrib.get("units")

        if pressure_pid is None and tau_pid is None:
            continue

        for dp in list(dg):
            if _strip_ns(dp.tag) != "dataPoint":
                continue

            p_el = None
            tau_el = None
            p_val = None
            tau_val = None

            for val_el in list(dp):
                xid = _strip_ns(val_el.tag)
                if pressure_pid is not None and xid == pressure_pid:
                    p_el = val_el
                    p_val = _as_number(val_el.text)
                elif tau_pid is not None and xid == tau_pid:
                    tau_el = val_el
                    tau_val = _as_number(val_el.text)

            p_input_units = pressure_units_src if pressure_units_src else None
            tau_input_units = tau_units_src if tau_units_src else None
            p_input_val = p_val if (p_val is not None and p_input_units) else None
            tau_input_val = tau_val if (tau_val is not None and tau_input_units) else None

            p_conv, p_units_conv, _, _, tau_conv, tau_units_conv = cantera_related_functions.convert_units(
                p_input_val,
                p_input_units,
                None,
                None,
                tau_input_val,
                tau_input_units,
            )

            if p_el is not None and p_conv is not None:
                p_el.text = f"{float(p_conv):.12g}"
            if tau_el is not None and tau_conv is not None:
                tau_el.text = f"{float(tau_conv):.12g}"

            if p_units_conv:
                pressure_units_out = p_units_conv
            if tau_units_conv:
                tau_units_out = tau_units_conv

        for ch in list(dg):
            if _strip_ns(ch.tag) != "property":
                continue
            pid = ch.attrib.get("id")
            if pid == pressure_pid and pressure_units_out:
                ch.attrib["units"] = pressure_units_out
            if pid == tau_pid and tau_units_out:
                ch.attrib["units"] = tau_units_out

    out_path = output_folder / xml_path.name
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def extract_idt_data_to_dataframe(folder_path: Union[str, Path], mechanism: str = "gri30.yaml") -> Any:
    """
    Extract IDT data from all XML files in a folder into a pandas DataFrame.
    
    Args:
        folder_path: Path to folder containing XML files
        mechanism: Cantera mechanism file for phi calculation (default: gri30.yaml)
    
    Returns a DataFrame with columns: T5, P5, composition, phi, tau, ignition_type, filename
    where T5 is temperature, P5 is pressure, tau is ignition delay time, and ignition_type
    is formatted as 'target;type'.
    """
    import pandas as pd  # type: ignore
    
    folder_path = Path(folder_path)
    all_data = []
    
    for file_path in folder_path.glob("*.xml"):
        try:
            exp = parse_idt_xml(file_path)
            composition_str = format_composition(exp.initial_composition)
            
            # Format ignition type
            ignition_type_str = ""
            ignition_amount = ""
            ignition_units = ""
            if exp.ignition_type:
                target = exp.ignition_type.target or ""              
                type_val = exp.ignition_type.type or ""
                ignition_amount = exp.ignition_type.amount if exp.ignition_type.amount is not None else ""
                ignition_units = exp.ignition_type.units if exp.ignition_type.units is not None else ""
                ignition_type_str = f"{target}{type_val}"
            
            # Calculate phi using Cantera
            phi = calculate_phi(exp.initial_composition)
            short_author = _author_before_first_comma(exp.bibliography_author)
            short_author = _clean_author_letters_only(short_author)
            year_str = (exp.bibliography_year or "").strip()
            author_year = f"{short_author}{year_str}" if short_author and year_str else (short_author or year_str)
            
            # Extract units from data group properties
            T_units = ""
            P_units = ""
            Tau_units = ""
            dPdt_units = ""
            for group in exp.data_groups:
                for prop_id, prop_def in group.properties.items():
                    if prop_def.name.lower() == "temperature" or (prop_def.label and prop_def.label.lower() == "t"):
                        T_units = prop_def.units or T_units
                    elif prop_def.name.lower() == "pressure" or (prop_def.label and prop_def.label.lower() == "p"):
                        P_units = prop_def.units or P_units
                    elif prop_def.name.lower() == "ignition delay" or (prop_def.label and prop_def.label.lower() == "tau"):
                        Tau_units = prop_def.units or Tau_units
                    elif prop_def.name.lower() == "pressure rise" or (prop_def.label and prop_def.label.lower() == "dp/dt"):
                        dPdt_units = prop_def.units or dPdt_units
            if not T_units:
                T_units = exp.common_property_units.get("temperature") or ""
            if not P_units:
                P_units = exp.common_property_units.get("pressure") or ""
            # Extract data from all data groups
            for group in exp.data_groups:
                for row in group.rows:
                    # Look for temperature (T), pressure (P), and ignition delay (tau)
                    # Use common properties if not in row, otherwise use row values
                    T = row.get("T") or row.get("temperature") or row.get("Temperature") or exp.common_properties.get("temperature")
                    P = row.get("P") or row.get("pressure") or row.get("Pressure") or exp.common_properties.get("pressure")
                    Tau = row.get("tau") or row.get("ignition delay") or row.get("Ignition Delay")
                    dPdt = row.get("dP/dt") or row.get("pressure rise") or row.get("Pressure rise")

                    P_conv, P_units_conv, ignition_amount_conv, ignition_units_conv, Tau_conv, Tau_units_conv = cantera_related_functions.convert_units(
                        P, P_units, ignition_amount, ignition_units, Tau, Tau_units
                    )
                    
                    all_data.append({
                        "T5": T,
                        "T5_units": T_units,
                        "P5": P_conv,
                        "P5_units": P_units_conv,
                        "composition": composition_str,
                        "phi": phi,
                        "tau": Tau_conv,
                        "tau_units": Tau_units_conv,
                        "pressure_rise": dPdt,
                        "pressure_rise_units": dPdt_units,
                        "ignition_type": ignition_type_str,
                        "ignition_target": exp.ignition_type.target.upper(),
                        "ignition_type": exp.ignition_type.type,
                        "ignition_amount": ignition_amount_conv,
                        "ignition_units": ignition_units_conv,
                        "author_year": author_year,
                        "filename": file_path.name
                    })
        except Exception as e:
            print(f"Error parsing {file_path.name}: {e}")
    
    return pd.DataFrame(all_data)


if __name__ == "__main__":
    # # Example usage
    # folder_path = "data/idt_data/syngas"
    
    # # Create DataFrame with all IDT data
    # df = extract_idt_data_to_dataframe(folder_path)
    # df.to_csv("idt_data_summary.csv", index=False)
    # print("\nIDT Data Summary:")
    # print(df.head(10))
    # print(f"\nTotal data points: {len(df)}")

    folder_path = "data/idt_data/syngas"
    output_folder = "data/idt_data_modified/syngas_with_phi"
    # folder_path = "data/idt_data/hydrogen"
    # output_folder = "data/idt_data_modified/hydrogen_with_phi"
    # folder_path = "data/idt_data/ethylene"
    # output_folder = "data/idt_data_modified/ethylene_with_phi"
    for file_path in Path(folder_path).glob("*.xml"):
        try:
            out_path = add_phi_to_xml_copy(file_path, output_folder)
            print(f"Processed {file_path.name} -> {out_path.name}")
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")



