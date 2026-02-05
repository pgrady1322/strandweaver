"""
StrandWeaver v0.1.0

Sequence utility functions for StrandWeaver.

Provides common sequence manipulation and analysis functions.
"""

from typing import List, Dict, Tuple


def extract_kmers(sequence: str, k: int) -> List[str]:
    """
    Extract all k-mers from a sequence.
    
    Args:
        sequence: DNA sequence string
        k: K-mer size
        
    Returns:
        List of k-mer strings
        
    Example:
        >>> extract_kmers("ATCGATCG", 3)
        ['ATC', 'TCG', 'CGA', 'GAT', 'ATC', 'TCG']
    """
    if k > len(sequence):
        return []
    
    sequence = sequence.upper()
    kmers = []
    
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i + k])
    
    return kmers


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate GC content of a DNA sequence.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        GC content as fraction (0.0 to 1.0)
        
    Example:
        >>> calculate_gc_content("ATGC")
        0.5
    """
    if not sequence:
        return 0.0
    
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    
    return gc_count / len(sequence)


def reverse_complement(sequence: str) -> str:
    """
    Generate reverse complement of DNA sequence.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Reverse complement sequence
        
    Example:
        >>> reverse_complement("ATCG")
        'CGAT'
    """
    complement_map = {
        'A': 'T', 'T': 'A',
        'G': 'C', 'C': 'G',
        'N': 'N',
        'a': 't', 't': 'a',
        'g': 'c', 'c': 'g',
        'n': 'n'
    }
    
    return ''.join(complement_map.get(base, base) for base in reversed(sequence))


def count_homopolymers(sequence: str, min_length: int = 4) -> List[Dict[str, any]]:
    """
    Identify homopolymer runs in sequence.
    
    Args:
        sequence: DNA sequence string
        min_length: Minimum homopolymer length to report
        
    Returns:
        List of dicts with 'base', 'position', 'length'
        
    Example:
        >>> count_homopolymers("ATCGAAAAGTC", min_length=4)
        [{'base': 'A', 'position': 4, 'length': 5}]
    """
    if not sequence:
        return []
    
    sequence = sequence.upper()
    homopolymers = []
    
    i = 0
    while i < len(sequence):
        current_base = sequence[i]
        length = 1
        
        # Count consecutive identical bases
        while i + length < len(sequence) and sequence[i + length] == current_base:
            length += 1
        
        # Report if meets minimum length
        if length >= min_length and current_base in 'ATGC':
            homopolymers.append({
                'base': current_base,
                'position': i,
                'length': length
            })
        
        i += length
    
    return homopolymers


__all__ = [
    'extract_kmers',
    'calculate_gc_content',
    'reverse_complement',
    'count_homopolymers'
]
