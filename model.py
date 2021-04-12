import torch
import time

from .helper_func import flat_accuracy, format_time


from transformers import get_linear_schedule_with_warmup, AdamW

def train_model(model, device, train_dataloader, val_dataloader, epochs=5, batch_to_print=100):
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    # Initialize scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Record history
    training_stats = []
    # Measuer the total training time
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):
        print("==== Epoch {:} / {:} ================================".format(epochs, epochs))
        print("Training started...")

        # Measure training time for a epoch
        t0 = time.time()

        # Reset the total loss for this epoch
        total_train_loss = 0

        # Switch to train mode
        model.train()

        # Iterate through all training batches
        for step, batch in enumerate(train_dataloader):
            if step % batch_to_print == 0 and step != 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                # evaluate_model(model, device, val_dataloader)
        
            # Unpack training data
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Clear previously calculated gradients
            model.zero_grad()

            # Forward pass
            result = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
                return_dict=True)
            
            loss = result.loss
            logits = result.logits
            # Add loss
            total_train_loss += loss.item()
            # Perform a backward pass
            loss.backward()
            # Clip the norm of the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update weights
            optimizer.step()
            # Update learning rate
            scheduler.step()
        
        # Calculate the average loss
        avg_train_loss = total_train_loss / len(train_dataloader)
        # Measure training time
        training_time = format_time(time.time() - t0)

        print()
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # Validation
        print()
        print("Validation started...")
        
        val_res = evaluate_model(model, device, val_dataloader)

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'training_loss': avg_train_loss,
                'valid_loss': val_res[1],
                'valid_acc': val_res[0],
                'training_time': training_time,
                'valid_time': val_res[2]
            }
        )
        return training_stats

def evaluate_model(model, device, eval_dataloader):
    print("Evaluation started...")
    t0 = time.time()
    model.eval()
    # Validation parameters
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Iterate through all eval batches
    for step, batch in enumerate(eval_dataloader):
        # Unpack data
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Don't need to store gradients values
        with torch.no_grad():
            result = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
                return_dict=True)

        # Calculate loss and logits
        # Loss for average loss, logits for accuracy
        loss = result.loss
        logits = result.logits

        total_eval_loss += loss.item()
        
        # Move logits and labels back to cpu
        # OK, but why
        # Why these two varialbes need different function?
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate accuracy
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    
    avg_eval_acc = total_eval_accuracy / len(eval_dataloader)
    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_eval_loss))
    print("  Validation Accuracy: {0:.2f}".format(avg_eval_acc))
    print("  Validation took: {:}".format(validation_time))

    model.train()
    return (avg_eval_acc, avg_eval_loss, validation_time)