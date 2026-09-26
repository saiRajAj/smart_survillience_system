from risk.risk_engine import RiskEngine
def test_low():assert RiskEngine().evaluate({'speed':0,'acceleration':0,'wrist_speed':0})['severity']=='LOW'
def test_high():assert RiskEngine().evaluate({'speed':300,'acceleration':400,'wrist_speed':500})['score']>=60
