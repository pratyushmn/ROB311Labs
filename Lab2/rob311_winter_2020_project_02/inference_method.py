from collections import deque
from support import definite_clause

### THIS IS THE TEMPLATE FILE
### WARNING: DO NOT CHANGE THE NAME OF FILE OR THE FUNCTION SIGNATURE


def pl_fc_entails(symbols_list : list, KB_clauses : list, known_symbols : list, query : int) -> bool:
    """
    pl_fc_entails function executes the Propositional Logic forward chaining algorithm (AIMA pg 258).
    It verifies whether the Knowledge Base (KB) entails the query
        Inputs
        ---------
            symbols_list  - a list of symbol(s) (have to be integers) used for this inference problem
            KB_clauses    - a list of definite_clause(s) composed using the numbers present in symbols_list
            known_symbols - a list of symbol(s) from the symbols_list that are known to be true in the KB (facts)
            query         - a single symbol that needs to be inferred

            Note: Definitely check out the test below. It will clarify a lot of your questions.

        Outputs
        ---------
        return - boolean value indicating whether KB entails the query
    """

    ### START: Your code

    inferred = [False for i in range(max(symbols_list) + 1)] # initially, assume all symbols are false unless proven otherwise

    # store the number of symbols (which we think are false) in each clause - start by assuming all symbols are false
    count = dict() 
    for clause in KB_clauses:
        count[clause] = len(clause.body)

    # go through all known symbols, and add more known symbols if a clause is true - ie. the conclusion of the clause must then also be true
    while len(known_symbols) != 0:
        p = known_symbols.pop()

        # if a symbol from the known symbols list is the query, then the query is entailed by the Knowledge Base
        if p == query:
            return True
        
        # if we haven't seen the symbol yet (ie. it shows false) then we need to update our knowledge since we now know that it is true
        # we also need to consider the fact that this symbol is true in every clause that contains it and update the counts
        if inferred[p] == False:
            inferred[p] = True

            # go through every c
            for clause in KB_clauses:
                if p in clause.body:
                    count[clause] = count[clause] - 1
                    if count[clause] == 0:
                        known_symbols.append(clause.conclusion) # if all the symbols in the clause are true, then the conclusion of the clause now becomes a new known symbol we need to explore

    return False 
    ### END: Your code


# SAMPLE TEST
if __name__ == '__main__':

    # Symbols used in this inference problem (Has to be Integers)
    symbols = [1,2,9,4,5]

    # Clause a: 1 and 2 => 9
    # Clause b: 9 and 4 => 5
    # Clause c: 1 => 4
    KB = [definite_clause([1, 2], 9), definite_clause([9,4], 5), definite_clause([1], 4)]

    # Known Symbols 1, 2
    known_symbols = [1, 2]

    # Does KB entail 5?
    entails = pl_fc_entails(symbols, KB, known_symbols, 5)

    print("Sample Test: " + ("Passed" if entails == True else "Failed"))
