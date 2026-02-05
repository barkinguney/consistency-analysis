from rmgpy.data.rmg import RMGDatabase
from rmgpy.species import Species
from rmgpy.molecule import Molecule
from pathlib import Path

import os
import pandas as pd

import xml.etree.ElementTree as ET


def find_reaction_in_all_libraries(database, reactant_smiles, product_smiles):
    """
    Search for a reaction across all loaded kinetics libraries.
    
    Args:
        database: RMGDatabase object
        reactant_smiles: List of SMILES strings for reactants
        product_smiles: List of SMILES strings for products
    
    Returns:
        Dictionary with library names as keys and matching reactions as values
    """
    # Create species from SMILES
    reactants = [Species(molecule=[Molecule(smiles=s)]) for s in reactant_smiles]
    products = [Species(molecule=[Molecule(smiles=s)]) for s in product_smiles]
    
    all_matches = {}
    
    # Iterate through all loaded libraries
    for library_name, library in database.kinetics.libraries.items():
        matches = []
        
        for index, entry in library.entries.items():
            reaction = entry.item
            
            # Check if reactants match (order-independent)
            if len(reaction.reactants) != len(reactants):
                continue
                
            reactants_match = all(
                any(r_search.is_isomorphic(r_rxn) for r_rxn in reaction.reactants)
                for r_search in reactants
            )
            
            # Check if products match (order-independent)
            if len(reaction.products) != len(products):
                continue
                
            products_match = all(
                any(p_search.is_isomorphic(p_rxn) for p_rxn in reaction.products)
                for p_search in products
            )
            
            if reactants_match and products_match:
                matches.append({
                    'reaction': reaction,
                    'kinetics': entry.data,
                    'index': index,
                    'entry': entry
                })
        
        if matches:
            all_matches[library_name] = matches
    
    return all_matches

def reaction_string_to_species_names(reaction_string):
    reaction_string = reaction_string.replace("<", " ")
    reaction_string = reaction_string.replace(">", " ")
    reaction_string = reaction_string.replace(" + ", " ")
    reactants_str, products_str = reaction_string.split("=")
    reactants = [s for s in reactants_str.split()]
    products = [s for s in products_str.split()]
    return reactants, products

def get_smiles_from_name(xml_root, preferred_key):
    for species in xml_root.findall("species"):
        if species.get("preferredKey") == preferred_key:
            smiles = species.find("SMILES")
            return smiles.text if smiles is not None else None
    return None



database = RMGDatabase()
database.load(
    "/home/barkin/RMG-database/input",
    kinetics_families="none",
    reaction_libraries=None    
)


important_reactions = [ "CO + OH <=> CO2 + H", "HO2 + H <=> OH + OH"]

out_dir = Path("arrhenius_uncertainty/results")
out_dir.mkdir(parents=True, exist_ok=True)


def within_tol(a, b, tol=0.01):
    return abs(a - b) / abs(b) <= tol

for reaction_name in important_reactions:
    
    reactant, product = reaction_string_to_species_names(reaction_name)
    print(reactant, product)

    tree = ET.parse("speciesDatabase.xml")
    root = tree.getroot()

    reactant_smiles = [get_smiles_from_name(root, name) for name in reactant]
    product_smiles = [get_smiles_from_name(root, name) for name in product] 
    print(reactant_smiles, product_smiles)



    results = find_reaction_in_all_libraries(
        database,
        reactant_smiles=reactant_smiles,
        product_smiles=product_smiles
    )

    temp_grid = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300]

    # Display results from all libraries
    print(f"Found reaction in {len(results)} libraries:\n")

    arrhenius_param_rows = []
    rate_coeff_rows = []

    for library_name, matches in results.items():
        print(f"\n{'='*60}")
        print(f"Library: {library_name}")
        print(f"{'='*60}")
        
        for i, match in enumerate(matches):
            reaction = match['reaction']
            kinetics = match['kinetics']
            
            print(f"\n  Match {i+1}: {reaction}")
            print(f"  Kinetics type: {type(kinetics).__name__}")     
            if type(kinetics).__name__ == "Arrhenius":

                current_params = {
                    'A': kinetics.A.value_si if kinetics.A else None,
                    'n':kinetics.n.value_si if kinetics.n else None,
                    'Ea':kinetics.Ea.value_si if kinetics.Ea else None,
                    'T0':kinetics.T0.value_si if kinetics.T0 else None,
                    'Tmin':kinetics.Tmin.value_si if kinetics.Tmin else None,
                    'Tmax':kinetics.Tmax.value_si if kinetics.Tmax else None,
                    'Pmin':kinetics.Pmin.value_si if kinetics.Pmin else None,
                    'Pmax':kinetics.Pmax.value_si if kinetics.Pmax else None,
                    'Type': type(kinetics).__name__,
                    'Library': library_name,
                    'comment':getattr(kinetics, 'comment', '')
                }
                duplicate = False
                for row in arrhenius_param_rows:
                    if (row['Type'] == current_params['Type'] and
                        within_tol(row['A'], current_params['A']) and
                        within_tol(row['n'], current_params['n']) and
                        within_tol(row['Ea'], current_params['Ea']) and
                        row['Tmin'] == current_params['Tmin'] and
                        row['Tmax'] == current_params['Tmax'] 
                    ):
                        duplicate = True
                        break
                if duplicate:
                    continue
                else:
                    arrhenius_param_rows.append(current_params)
                    reaction_temp_grid = temp_grid
                    for T in reaction_temp_grid:
                        k = kinetics.get_rate_coefficient(T)
                        rate_coeff_rows.append({
                            'Temperature (K)': T,
                            'Rate Coefficient (m^3/(mol*s))': k,
                            'Library': library_name, 
                            'Reaction': reaction_name
                        })

                print(f"  Arrhenius coefficients:")
                print(f"    A  = {kinetics.A.value_si:e} {kinetics.A.units}")
                print(f"    n  = {kinetics.n.value_si}")
                print(f"    Ea = {kinetics.Ea.value_si/1000:.2f} kJ/mol")
                print(f"    Tmin = {kinetics.Tmin} K")
                print(f"    Tmax = {kinetics.Tmax} K")
                
                # Calculate rate at 1000 K for comparison
                k_1000 = kinetics.get_rate_coefficient(1000)
                print(f"    k(1000 K) = {k_1000:e}")
            elif type(kinetics).__name__ == "MultiArrhenius":
                current_rows = []
                for arr in kinetics.arrhenius:
                    current_params = {
                        'A': arr.A.value_si if arr.A else None,
                        'n':arr.n.value_si if arr.n else None,
                        'Ea':arr.Ea.value_si if arr.Ea else None,
                        'T0':arr.T0.value_si if arr.T0 else None,
                        'Tmin':arr.Tmin.value_si if arr.Tmin else None,
                        'Tmax':arr.Tmax.value_si if arr.Tmax else None,
                        'Pmin':arr.Pmin.value_si if arr.Pmin else None,
                        'Pmax':arr.Pmax.value_si if arr.Pmax else None,
                        'Type': type(kinetics).__name__,
                        'Library': library_name,
                        'comment':getattr(arr, 'comment', '')
                    }
                    current_rows.append(current_params)
                duplicate = True
                for row in current_rows:
                    is_duplicate = False
                    for existing_row in arrhenius_param_rows:
                        if (existing_row['Type'] == row['Type'] and
                            within_tol(existing_row['A'], row['A']) and
                            within_tol(existing_row['n'], row['n']) and
                            within_tol(existing_row['Ea'], row['Ea']) and
                            existing_row['Tmin'] == row['Tmin'] and
                            existing_row['Tmax'] == row['Tmax'] 
                        ):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        duplicate = False
                        break   
                if duplicate:
                    continue
                else:
                    arrhenius_param_rows.extend(current_rows)
                    reaction_temp_grid = temp_grid
                    for T in reaction_temp_grid:
                        k = kinetics.get_rate_coefficient(T)
                        rate_coeff_rows.append({
                            'Temperature (K)': T,
                            'Rate Coefficient (m^3/(mol*s))': k,
                            'Library': library_name, 
                            'Reaction': reaction_name
                        })
            elif type(kinetics).__name__ == "PDepArrhenius":
                pass
            elif type(kinetics).__name__ == "ThirdBody":
                pass
            else:
                print(f" Unsupported kinetics type {type(kinetics).__name__}")
    
    
    rate_coeff_dfs = pd.DataFrame(rate_coeff_rows)
    rate_coeff_dfs.to_csv(f"{out_dir}/rate_coefficients_{reaction_name}.csv", index=False)
    arrhenius_param_dfs = pd.DataFrame(arrhenius_param_rows)
    arrhenius_param_dfs.to_csv(f"{out_dir}/arrhenius_params_{reaction_name}.csv", index=False)
    



