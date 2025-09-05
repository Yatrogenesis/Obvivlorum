import asyncio 
from ai_simple_working import SimpleWorkingAI 
 
ai = SimpleWorkingAI() 
 
async def test_ai(): 
    print("=== PROBANDO AI INTELIGENTE ===") 
    questions = [ 
        "Que es la inteligencia artificial", 
        "Como funciona una red neuronal", 
        "Hola", 
        "Cuales son tus capacidades", 
        "Por que es importante la ciberseguridad" 
    ] 
 
    for q in questions: 
        print(f"\nPregunta: {q}") 
        response = await ai.process_message(q) 
        print(f"Respuesta: {response[:200]}...") 
 
if __name__ == "__main__": 
    asyncio.run(test_ai()) 
