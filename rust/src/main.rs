#[derive(Debug)]
pub enum Op {
    Add(Box<Op>, Box<Op>),
    Sub(Box<Op>, Box<Op>),
    Mul(Box<Op>, Box<Op>),
    Div(Box<Op>, Box<Op>),
    // eventually add to grammar
    // Min(Box<Op>, Box<Op>),
    // Max(Box<Op>, Box<Op>),
    Exp(Box<Op>, Box<Op>),
    Variable(String),
    Constant(f32),
}

// simple recursive descent on expression
// should add parens to force binding
fn parse_exp(s: &String, start: usize) -> (Box<Op>, usize) {
    println!("Calling parse on {}:{}", s, start);
    // no error handling currently
    let mut end = start;

    let mut c = s.as_bytes()[end];
    let mut acc = String::from("");
    while c.is_ascii_alphabetic() {
        acc += (c as char).to_string().as_str();
        end += 1;
        if end >= s.len() {
            break;
        }
        c = s.as_bytes()[end];
    }

    let terminal: Box<Op>;

    // first value must be constant or variable
    if acc.len() != 0 {
        terminal = Box::new(Op::Variable(acc));
    } else {
        while c.is_ascii_digit() {
            acc += (c as char).to_string().as_str();
            end += 1;
            if end >= s.len() {
                break;
            }
            c = s.as_bytes()[end];
        }
        let val: f32 = acc.parse().unwrap();
        terminal = Box::new(Op::Constant(val));
    }

    // base case
    if end >= s.len() {
        return (terminal, end);
    }

    // general case
    let op_c = s.as_bytes()[end];
    let (right, right_end) = parse_exp(&s, end + 1);

    // parse op and construct
    let op = match op_c as char {
        '+' => Op::Add(terminal, right),
        '-' => Op::Sub(terminal, right),
        '*' => Op::Mul(terminal, right),
        '/' => Op::Div(terminal, right),
        '^' => Op::Exp(terminal, right), // this is not actually what exp should be, that's a unary
        // op
        _ => panic!("not supported op"),
    };

    return (Box::new(op), right_end);
}

fn main() {
    let (ast, _) = parse_exp(&String::from("1^x+3555*long"), 0);
    println!("ast: {:?}", ast);
    println!("Hello, world!");
}
